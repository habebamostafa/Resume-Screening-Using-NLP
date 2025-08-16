import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import re
import spacy
from PyPDF2 import PdfReader
import docx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os 
# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Page configuration
st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .st-b7 {
        color: white;
    }
    .st-cb {
        background-color: #f0f2f6;
    }
    .css-1aumxhk {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

class ResumeScreener:
    def __init__(self):
        self.model = None
        self.df = None
        self.evaluation_metrics = {}
        self.is_trained = False

    def load_data(self, file_path):
        """Load and preprocess data"""
        self.df = pd.read_csv(file_path)
        self.df['job_description'] = self.df['job_description'].apply(self.clean_text)
        self.df['resume'] = self.df['resume'].apply(self.clean_text)
        return self.df

    def clean_text(self, text):
        """Clean text data"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def train_model(self):
        """Train the recommendation model"""
        # Split data
        X = self.df[['job_description', 'resume']]
        y = self.df['match_score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create features
        X_train_features = self.create_features(X_train)
        X_test_features = self.create_features(X_test)

        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_features, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_features)
        test_pred = self.model.predict(X_test_features)

        self.evaluation_metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }

        # Save model
        joblib.dump(self.model, 'resume_screener_model.joblib')
        self.is_trained = True

        return self.evaluation_metrics

    def create_features(self, X):
        """Create features from text data"""
        # Text embeddings
        job_embeddings = embedding_model.encode(X['job_description'])
        resume_embeddings = embedding_model.encode(X['resume'])
        
        # Cosine similarities
        similarities = np.array([cosine_similarity([je], [re])[0][0] 
                             for je, re in zip(job_embeddings, resume_embeddings)])
        
        # Combine features
        features = np.hstack([job_embeddings, resume_embeddings, similarities.reshape(-1, 1)])
        return features

    def predict_match(self, job_desc, resume_text):
        """Predict match score between job and resume"""
        if not self.is_trained:
            try:
                self.model = joblib.load('resume_screener_model.joblib')
                self.is_trained = True
            except:
                raise Exception("Model not trained. Please train the model first.")
        
        # Clean texts
        job_clean = self.clean_text(job_desc)
        resume_clean = self.clean_text(resume_text)
        
        # Create features
        X = pd.DataFrame({'job_description': [job_clean], 'resume': [resume_clean]})
        features = self.create_features(X)
        
        # Predict
        prediction = self.model.predict(features)[0]
        return min(max(prediction, 1), 5)  # Ensure score is between 1-5

    def extract_skills(self, text):
        """Extract skills from text"""
        doc = nlp(text)
        skills = set()
        
        # Pattern matching for skills
        skill_patterns = [
            r'\bpython\b', r'\bsql\b', r'\bexcel\b', 
            r'\bmachine learning\b', r'\bdata analysis\b',
            r'\btableau\b', r'\bpower bi\b', r'\bstatistics\b',
            r'\bdeep learning\b', r'\bnlp\b'
        ]
        
        for pattern in skill_patterns:
            if re.search(pattern, text):
                skills.add(pattern.strip('\\b'))
        
        # NER for skills
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'TECH']:
                skills.add(ent.text.lower())
        
        return sorted(skills)

    def read_file(self, file):
        """Read uploaded file"""
        if file.type == "application/pdf":
            reader = PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages])
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            text = " ".join([para.text for para in doc.paragraphs])
        else:
            text = str(file.read(), "utf-8")
        return text

# Main application
def main():
    st.title("📄 AI-Powered Resume Screening System")
    st.markdown("Upload a job description and resume to get compatibility score")
    
    # Initialize screener
    screener = ResumeScreener()
    
    # Load data (in a real app, you'd upload this or have it pre-loaded)
    data_path = "resume_job_dataset.csv"  # Change to your dataset path
    try:
        screener.load_data(data_path)
    except:
        st.error("Could not load dataset. Please check the file path.")
        return
    
    # Sidebar for model training
    with st.sidebar:
        st.header("Model Training")
        
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes"):
                metrics = screener.train_model()
                st.success("Model trained successfully!")
                
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                col1.metric("Train RMSE", f"{metrics['train_rmse']:.2f}")
                col2.metric("Test RMSE", f"{metrics['test_rmse']:.2f}")
                col1.metric("Train R²", f"{metrics['train_r2']:.2f}")
                col2.metric("Test R²", f"{metrics['test_r2']:.2f}")
                
                # Plot feature importance
                if hasattr(screener.model, 'feature_importances_'):
                    fig, ax = plt.subplots()
                    ax.barh(range(10), screener.model.feature_importances_[:10])
                    ax.set_title("Top 10 Important Features")
                    st.pyplot(fig)
    
    # Main content
    tab1, tab2 = st.tabs(["Single Evaluation", "Batch Processing"])
    
    with tab1:
        st.header("Evaluate Single Resume")
        
        # Job description input
        job_desc = st.text_area("Paste Job Description Here", height=200,
                               placeholder="Enter the job description including required skills...")
        
        # Resume upload
        uploaded_file = st.file_uploader("Upload Resume File", 
                                       type=["pdf", "docx", "txt"],
                                       help="Upload PDF, Word, or Text file")
        
        if uploaded_file and job_desc:
            with st.spinner("Analyzing resume..."):
                try:
                    # Read resume
                    resume_text = screener.read_file(uploaded_file)
                    
                    # Extract skills
                    job_skills = screener.extract_skills(job_desc)
                    resume_skills = screener.extract_skills(resume_text)
                    matching_skills = set(job_skills).intersection(set(resume_skills))
                    
                    # Predict match
                    match_score = screener.predict_match(job_desc, resume_text)
                    normalized_score = (match_score - 1) / 4  # Normalize to 0-1
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Score with progress bar
                    col1, col2 = st.columns([1, 3])
                    col1.metric("Match Score", f"{match_score:.1f}/5")
                    col2.progress(normalized_score)
                    
                    # Skills analysis
                    with st.expander("Skills Analysis", expanded=True):
                        st.write(f"**{len(matching_skills)}/{len(job_skills)}** required skills matched")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Matching Skills**")
                            if matching_skills:
                                st.write(", ".join(matching_skills))
                            else:
                                st.warning("No matching skills found")
                        
                        with col2:
                            st.markdown("**Missing Skills**")
                            missing_skills = set(job_skills) - set(resume_skills)
                            if missing_skills:
                                st.write(", ".join(missing_skills))
                            else:
                                st.success("All required skills present!")
                    
                    # Recommendation
                    st.subheader("Recommendation")
                    if match_score >= 4:
                        st.success("Strong Match - Highly recommended for this position")
                    elif match_score >= 3:
                        st.info("Good Match - Candidate meets most requirements")
                    elif match_score >= 2:
                        st.warning("Partial Match - Some key skills missing")
                    else:
                        st.error("Weak Match - Does not meet most requirements")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.header("Batch Processing")
        st.info("Coming soon - Upload multiple resumes to evaluate against one job description")
        # Future implementation would go here

if __name__ == "__main__":
    main()