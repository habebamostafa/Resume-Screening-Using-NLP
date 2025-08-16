import numpy as np
import pandas as pd
import re
import spacy
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import xgboost as xgb
from PyPDF2 import PdfReader
from docx import Document

nltk.download('punkt')
nltk.download('stopwords')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM=384

class ResumeScreener:
    def __init__(self):
        self.model = None
        self.embedding_model = embedding_model
        self.expected_feature_dim = None
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_text_from_file(self, file_path):
        """Extract text from different file formats"""
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = " ".join([page.extract_text() for page in reader.pages])
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file format")
        return self.clean_text(text)
    
    def extract_skills(self, text):
        """Extract skills from text using patterns and NER"""
        doc = nlp(text)
        skills = set()
        
        # Skill patterns with clean output names
        skill_patterns = [
            (r'\bpython\b', 'python'),
            (r'\bsql\b', 'sql'),
            (r'\bexcel\b', 'excel'), 
            (r'\bmachine[\s-]?learning\b', 'machine learning'),
            (r'\bdata[\s-]?analysis\b', 'data analysis'),
            (r'\btableau\b', 'tableau'),
            (r'\bpower[\s-]?bi\b', 'power bi'),
            (r'\bstatistics\b', 'statistics'),
            (r'\bdeep[\s-]?learning\b', 'deep learning'),
            (r'\bnlp\b', 'nlp'),
            (r'\bjava\b', 'java'),
            (r'\bc\+\+\b', 'c++'),
            (r'\bjavascript\b', 'javascript'),
            (r'\bhtml\b', 'html'),
            (r'\bcss\b', 'css'),
            (r'\baws\b', 'aws'),
            (r'\bazure\b', 'azure'),
            (r'\bgoogle[\s-]?cloud\b', 'google cloud'),
            (r'\btensorflow\b', 'tensorflow'),
            (r'\bpytorch\b', 'pytorch'),
            (r'\bscikit[\s-]?learn\b', 'scikit-learn')
        ]
        
        for pattern, clean_name in skill_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                skills.add(clean_name)
        
        # Extract skills using NER
        for ent in doc.ents:
            if ent.label_ == "SKILL" or ent.label_ == "TECH":
                skills.add(ent.text.lower())
        
        return sorted(skills)
    
    def create_features(self, X):
        """Create feature vectors from job descriptions and resumes"""
        job_embeddings = self.embedding_model.encode(
            X['job_description'].tolist(), 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        resume_embeddings = self.embedding_model.encode(
            X['resume'].tolist(), 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Calculate cosine similarities
        similarities = np.array([cosine_similarity([je], [re])[0][0] 
                         for je, re in zip(job_embeddings, resume_embeddings)])
        
        # Combine features
        features = np.hstack([
            job_embeddings, 
            resume_embeddings, 
            similarities.reshape(-1, 1)
        ])
        
        return features
    
    def train_model(self, data_path):
        """Train the matching model"""
        df = pd.read_csv(data_path)
        df['job_description'] = df['job_description'].apply(self.clean_text)
        df['resume'] = df['resume'].apply(self.clean_text)
        
        X = df[['job_description', 'resume']]
        y = df['match_score']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Encoding training data...")
        X_train_features = self.create_features(X_train)
        print("Encoding testing data...")
        X_test_features = self.create_features(X_test)
        
        # Store the expected feature dimension
        self.expected_feature_dim = X_train_features.shape[1]
        
        # Updated XGBoost initialization without deprecated parameters
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            tree_method="hist",
            device="cuda",  # Updated way to specify GPU usage
            random_state=42,
            enable_categorical=True
        )
        
        self.model.fit(X_train_features, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train_features)
        test_pred = self.model.predict(X_test_features)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
        
        return self.model
    
    def save_model(self, filepath):
        """Save trained model to file"""
        save_data = {
            'model': self.model,
            'expected_feature_dim': self.expected_feature_dim
        }
        joblib.dump(save_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        loaded_data = joblib.load(filepath)
        self.model = loaded_data['model']
        self.expected_feature_dim = loaded_data['expected_feature_dim']
        print(f"Model loaded from {filepath}")
        return self.model
    
    def predict_match(self, job_desc, resume_text):
        """Predict match score between job description and resume"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
            
        job_clean = self.clean_text(job_desc)
        resume_clean = self.clean_text(resume_text)
        
        X_new = pd.DataFrame({
            'job_description': [job_clean],
            'resume': [resume_clean]
        })
        
        features = self.create_features(X_new)
        
        # Verify feature dimensions
        if features.shape[1] != self.expected_feature_dim:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.expected_feature_dim}, "
                f"got {features.shape[1]}. Did you change the embedding model?"
            )
        
        score = self.model.predict(features)[0]
        return min(max(score, 1), 5)  # Ensure score is between 1 and 5
    
    def analyze_match(self, job_desc, resume_text):
        """Provide detailed analysis of match between job and resume"""
        score = self.predict_match(job_desc, resume_text)
        job_skills = self.extract_skills(job_desc)
        resume_skills = self.extract_skills(resume_text)
        
        matching_skills = set(job_skills) & set(resume_skills)
        missing_skills = set(job_skills) - set(resume_skills)
        extra_skills = set(resume_skills) - set(job_skills)
        
        return {
            'match_score': score,
            'match_percentage': (score / 5) * 100,
            'matching_skills': sorted(matching_skills),
            'missing_skills': sorted(missing_skills),
            'extra_skills': sorted(extra_skills)
        }
