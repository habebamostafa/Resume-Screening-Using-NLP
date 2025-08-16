import re
import numpy as np
import pandas as pd
import spacy
import joblib
import xgboost as xgb
from PyPDF2 import PdfReader
import docx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pathlib import Path

class ResumeScreener:
    def __init__(self, model_path=None):
        """
        Initialize the ResumeScreener
        
        Args:
            model_path (str): Path to the pre-trained .joblib model file
        """
        # Initialize NLP components
        self.nlp = self._load_spacy_model()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Model loading
        self.model = None
        self.expected_feature_dim = None
        
        if model_path:
            self.load_model(model_path)

    def _load_spacy_model(self):
        try:
            return spacy.load("en_core_web_sm")
        except:
            # Fallback to English without model
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp

    def load_model(self, model_path):
        """
        Load pre-trained model from file
        
        Args:
            model_path (str): Path to the .joblib model file
        """
        try:
            # Verify model file exists
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Load model data
            loaded_data = joblib.load(model_path)
            
            # Validate loaded data
            if not all(key in loaded_data for key in ['model', 'expected_feature_dim']):
                raise ValueError("Invalid model file format")
            
            self.model = loaded_data['model']
            self.expected_feature_dim = loaded_data['expected_feature_dim']
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def extract_text_from_file(self, file_path):
        """Extract text from different file formats"""
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = " ".join([page.extract_text() for page in reader.pages])
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file format. Supported formats: PDF, DOCX, TXT")
        return self.clean_text(text)

    def extract_skills(self, text):
        """Extract skills from text using patterns and NER"""
        doc = self.nlp(text)
        skills = set()
        
        # Skill patterns with clean output names
        skill_patterns = [
            (r'\bpython\b', 'python'),
            (r'\bsql\b', 'sql'),
            # Add all your other skill patterns here...
        ]
        
        # Pattern matching
        for pattern, clean_name in skill_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                skills.add(clean_name)
        
        # NER extraction
        for ent in doc.ents:
            if ent.label_ in ["SKILL", "TECH"]:
                skills.add(ent.text.lower())
        
        return sorted(skills)

    def create_features(self, job_desc, resume_text):
        """
        Create feature vectors from job description and resume text
        
        Args:
            job_desc (str): Job description text
            resume_text (str): Resume text
            
        Returns:
            np.array: Feature vector for prediction
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first")
            
        # Get embeddings
        job_embedding = self.embedding_model.encode([job_desc], convert_to_numpy=True)[0]
        resume_embedding = self.embedding_model.encode([resume_text], convert_to_numpy=True)[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]
        
        # Combine features
        features = np.hstack([job_embedding, resume_embedding, similarity])
        
        # Verify dimensions
        if len(features) != self.expected_feature_dim:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.expected_feature_dim}, "
                f"got {len(features)}. Did you change the embedding model?"
            )
            
        return features.reshape(1, -1)

    def predict_match(self, job_desc, resume_text):
        """
        Predict match score between job description and resume
        
        Args:
            job_desc (str): Job description text
            resume_text (str): Resume text
            
        Returns:
            float: Match score between 1 and 5
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first")
            
        # Clean and prepare texts
        job_clean = self.clean_text(job_desc)
        resume_clean = self.clean_text(resume_text)
        
        # Create features and predict
        features = self.create_features(job_clean, resume_clean)
        score = self.model.predict(features)[0]
        
        # Ensure score is between 1 and 5
        return max(1, min(5, score))

    def analyze_match(self, job_desc, resume_text):
        """
        Comprehensive analysis of match between job and resume
        
        Args:
            job_desc (str): Job description text
            resume_text (str): Resume text
            
        Returns:
            dict: Analysis results including:
                - match_score
                - match_percentage
                - matching_skills
                - missing_skills
                - extra_skills
        """
        score = self.predict_match(job_desc, resume_text)
        job_skills = self.extract_skills(job_desc)
        resume_skills = self.extract_skills(resume_text)
        
        return {
            'match_score': score,
            'match_percentage': (score / 5) * 100,
            'matching_skills': sorted(set(job_skills) & set(resume_skills)),
            'missing_skills': sorted(set(job_skills) - set(resume_skills)),
            'extra_skills': sorted(set(resume_skills) - set(job_skills))
        }