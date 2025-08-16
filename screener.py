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
import streamlit as st
import io

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
            
            # Handle different model formats
            if isinstance(loaded_data, dict):
                # If it's a dictionary with model and dimensions
                if 'model' in loaded_data:
                    self.model = loaded_data['model']
                    self.expected_feature_dim = loaded_data.get('expected_feature_dim', 769)  # Default dimension
                else:
                    raise ValueError("Invalid model file format")
            else:
                # If it's just the model object
                self.model = loaded_data
                self.expected_feature_dim = 769  # Default dimension for MiniLM + similarity
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def extract_text_from_file(self, uploaded_file):
        """Extract text from different file formats - Streamlit compatible"""
        try:
            file_name = uploaded_file.name.lower()
            
            if file_name.endswith('.pdf'):
                # Read PDF using BytesIO
                pdf_bytes = uploaded_file.read()
                reader = PdfReader(io.BytesIO(pdf_bytes))
                text = " ".join([page.extract_text() for page in reader.pages])
                
            elif file_name.endswith('.docx'):
                # Read DOCX using BytesIO
                docx_bytes = uploaded_file.read()
                doc = docx.Document(io.BytesIO(docx_bytes))
                text = " ".join([para.text for para in doc.paragraphs])
                
            elif file_name.endswith('.txt'):
                # Read text file
                text = str(uploaded_file.read(), "utf-8")
                
            else:
                raise ValueError("Unsupported file format. Supported formats: PDF, DOCX, TXT")
                
            return self.clean_text(text)
            
        except Exception as e:
            raise ValueError(f"Error processing file: {str(e)}")

    def extract_skills(self, text):
        """Extract skills from text using comprehensive patterns"""
        text_lower = text.lower()
        skills = set()
        
        # Comprehensive skill patterns
        skill_patterns = {
            # Programming Languages
            r'\bpython\b': 'python',
            r'\bjava\b(?!\s*script)': 'java',
            r'\bjavascript\b|\bjs\b': 'javascript',
            r'\bc\+\+\b|\bcpp\b': 'c++',
            r'\bc#\b|\bc sharp\b': 'c#',
            r'\bphp\b': 'php',
            r'\bruby\b': 'ruby',
            r'\bgo\b|\bgolang\b': 'go',
            r'\brust\b': 'rust',
            r'\bswift\b': 'swift',
            r'\bkotlin\b': 'kotlin',
            r'\bscala\b': 'scala',
            r'\br\b(?=\s)': 'r',
            r'\bmatlab\b': 'matlab',
            
            # Web Technologies
            r'\bhtml\b|\bhtml5\b': 'html',
            r'\bcss\b|\bcss3\b': 'css',
            r'\breact\b|\breactjs\b': 'react',
            r'\bangular\b|\bangularjs\b': 'angular',
            r'\bvue\b|\bvuejs\b': 'vue',
            r'\bnode\b|\bnodejs\b': 'nodejs',
            r'\bexpress\b|\bexpressjs\b': 'express',
            r'\bdjango\b': 'django',
            r'\bflask\b': 'flask',
            r'\bspring\b|\bspring boot\b': 'spring',
            r'\bbootstrap\b': 'bootstrap',
            r'\bjquery\b': 'jquery',
            
            # Databases
            r'\bmysql\b': 'mysql',
            r'\bpostgresql\b|\bpostgres\b': 'postgresql',
            r'\bmongodb\b|\bmongo\b': 'mongodb',
            r'\boracle\b': 'oracle',
            r'\bsqlite\b': 'sqlite',
            r'\bredis\b': 'redis',
            r'\bcassandra\b': 'cassandra',
            r'\belasticsearch\b': 'elasticsearch',
            r'\bsql\b': 'sql',
            r'\bnosql\b': 'nosql',
            
            # Cloud & DevOps
            r'\baws\b|\bamazon web services\b': 'aws',
            r'\bazure\b|\bmicrosoft azure\b': 'azure',
            r'\bgcp\b|\bgoogle cloud\b': 'gcp',
            r'\bdocker\b': 'docker',
            r'\bkubernetes\b|\bk8s\b': 'kubernetes',
            r'\bterraform\b': 'terraform',
            r'\bansible\b': 'ansible',
            r'\bjenkins\b': 'jenkins',
            r'\bgit\b': 'git',
            r'\bgithub\b': 'github',
            r'\bgitlab\b': 'gitlab',
            r'\bci/cd\b|\bcicd\b': 'ci/cd',
            
            # Data Science & ML
            r'\bmachine learning\b|\bml\b': 'machine learning',
            r'\bdeep learning\b|\bdl\b': 'deep learning',
            r'\bartificial intelligence\b|\bai\b': 'artificial intelligence',
            r'\bdata science\b': 'data science',
            r'\bpandas\b': 'pandas',
            r'\bnumpy\b': 'numpy',
            r'\bscipy\b': 'scipy',
            r'\bscikit-learn\b|\bsklearn\b': 'scikit-learn',
            r'\btensorflow\b': 'tensorflow',
            r'\bpytorch\b': 'pytorch',
            r'\bkeras\b': 'keras',
            r'\btableau\b': 'tableau',
            r'\bpower bi\b|\bpowerbi\b': 'power bi',
            r'\br\b|\br programming\b': 'r',
            r'\bspss\b': 'spss',
            r'\bsas\b': 'sas',
            
            # Mobile Development
            r'\bandroid\b': 'android',
            r'\bios\b': 'ios',
            r'\breact native\b': 'react native',
            r'\bflutter\b': 'flutter',
            r'\bxamarin\b': 'xamarin',
            
            # Other Technologies
            r'\blinux\b': 'linux',
            r'\bwindows\b': 'windows',
            r'\bmacOS\b|\bmac os\b': 'macos',
            r'\bapi\b': 'api',
            r'\brest\b|\brestful\b': 'rest api',
            r'\bgraphql\b': 'graphql',
            r'\bmicroservices\b': 'microservices',
            r'\bagile\b': 'agile',
            r'\bscrum\b': 'scrum',
            r'\bkanban\b': 'kanban',
            r'\bjira\b': 'jira',
            r'\bconfluence\b': 'confluence',
            r'\bslack\b': 'slack',
            r'\bmicrosoft office\b|\bms office\b': 'microsoft office',
            r'\bexcel\b': 'excel',
            r'\bpowerpoint\b': 'powerpoint',
            r'\bword\b': 'word',
        }
        
        # Pattern matching
        for pattern, clean_name in skill_patterns.items():
            if re.search(pattern, text_lower):
                skills.add(clean_name)
        
        # NER extraction (if spacy model is available)
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "GPE"]:  # Organizations, products, locations
                    entity_text = ent.text.lower().strip()
                    if len(entity_text) > 2 and entity_text.isalpha():
                        skills.add(entity_text)
        except:
            pass
        
        return sorted(list(skills))

    def create_features(self, job_desc, resume_text):
        """
        Create feature vectors from job description and resume text
        
        Args:
            job_desc (str): Job description text
            resume_text (str): Resume text
            
        Returns:
            np.array: Feature vector for prediction
        """
        # Get embeddings
        job_embedding = self.embedding_model.encode([job_desc], convert_to_numpy=True)[0]
        resume_embedding = self.embedding_model.encode([resume_text], convert_to_numpy=True)[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]
        
        # Combine features
        features = np.hstack([job_embedding, resume_embedding, [similarity]])
        
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
        # If no model is loaded, use similarity-based scoring
        if not self.model:
            return self._similarity_based_score(job_desc, resume_text)
            
        try:
            # Clean and prepare texts
            job_clean = self.clean_text(job_desc)
            resume_clean = self.clean_text(resume_text)
            
            # Create features and predict
            features = self.create_features(job_clean, resume_clean)
            score = self.model.predict(features)[0]
            
            # Ensure score is between 1 and 5
            return max(1, min(5, float(score)))
            
        except Exception as e:
            # Fallback to similarity-based scoring
            print(f"Model prediction failed: {e}. Using similarity-based scoring.")
            return self._similarity_based_score(job_desc, resume_text)

    def _similarity_based_score(self, job_desc, resume_text):
        """Fallback scoring method based on text similarity and skill matching"""
        try:
            # Text similarity
            job_embedding = self.embedding_model.encode([job_desc])
            resume_embedding = self.embedding_model.encode([resume_text])
            text_similarity = cosine_similarity(job_embedding, resume_embedding)[0][0]
            
            # Skill matching
            job_skills = set(self.extract_skills(job_desc))
            resume_skills = set(self.extract_skills(resume_text))
            
            if len(job_skills) > 0:
                skill_match_ratio = len(job_skills.intersection(resume_skills)) / len(job_skills)
            else:
                skill_match_ratio = 0.5  # Neutral score if no skills detected
            
            # Combined score (weighted average)
            combined_score = (text_similarity * 0.6 + skill_match_ratio * 0.4) * 5
            
            return max(1, min(5, combined_score))
            
        except Exception as e:
            print(f"Similarity scoring failed: {e}. Using default score.")
            return 2.5  # Default middle score

    def analyze_match(self, job_desc, resume_text):
        """
        Comprehensive analysis of match between job and resume
        
        Args:
            job_desc (str): Job description text
            resume_text (str): Resume text
            
        Returns:
            dict: Analysis results
        """
        try:
            score = self.predict_match(job_desc, resume_text)
            job_skills = self.extract_skills(job_desc)
            resume_skills = self.extract_skills(resume_text)
            
            matching_skills = sorted(set(job_skills) & set(resume_skills))
            missing_skills = sorted(set(job_skills) - set(resume_skills))
            extra_skills = sorted(set(resume_skills) - set(job_skills))
            
            return {
                'match_score': round(score, 1),
                'match_percentage': round((score / 5) * 100, 1),
                'matching_skills': matching_skills,
                'missing_skills': missing_skills,
                'extra_skills': extra_skills,
                'total_job_skills': len(job_skills),
                'total_resume_skills': len(resume_skills),
                'skills_match_ratio': round(len(matching_skills) / max(len(job_skills), 1), 2)
            }
            
        except Exception as e:
            return {
                'match_score': 1.0,
                'match_percentage': 20.0,
                'matching_skills': [],
                'missing_skills': [],
                'extra_skills': [],
                'error': str(e)
            }