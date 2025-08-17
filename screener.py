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
        """Extract skills from text using comprehensive patterns and NER"""
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
            r'\btypescript\b': 'typescript',
            
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
            r'\bwebpack\b': 'webpack',
            r'\bbabel\b': 'babel',
            r'\bsass\b|\bscss\b': 'sass',
            
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
            r'\bfirebase\b': 'firebase',
            r'\bdynamodb\b': 'dynamodb',
            
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
            r'\bvagrant\b': 'vagrant',
            r'\bchef\b': 'chef',
            r'\bpuppet\b': 'puppet',
            
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
            r'\bjupyter\b': 'jupyter',
            r'\bmatplotlib\b': 'matplotlib',
            r'\bseaborn\b': 'seaborn',
            r'\bplotly\b': 'plotly',
            
            # Mobile Development
            r'\bandroid\b': 'android',
            r'\bios\b': 'ios',
            r'\breact native\b': 'react native',
            r'\bflutter\b': 'flutter',
            r'\bxamarin\b': 'xamarin',
            r'\bionic\b': 'ionic',
            r'\bcordova\b': 'cordova',
            
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

    def extract_named_entities(self, text):
        """Extract named entities from resume text"""
        entities = {
            'experience_level': self._extract_experience(text),
            'education': self._extract_education(text),
            'certifications': self._extract_certifications(text),
            'companies': self._extract_companies(text),
            'projects': self._extract_projects(text),
            'languages': self._extract_languages(text),
            'contact_info': self._extract_contact_info(text)
        }
        return entities

    def _extract_experience(self, text):
        """Extract experience level from text"""
        text_lower = text.lower()
        
        # Experience patterns
        exp_patterns = [
            r'(\d+)[\s\-]*years?\s+(?:of\s+)?experience',
            r'(\d+)\+\s*years',
            r'experience.*?(\d+)\s*years',
            r'(\d+)\s*years\s*(?:of\s+)?(?:professional\s+)?experience'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = match.group(1)
                return f"{years} years of experience"
        
        # Level indicators
        if re.search(r'\b(?:entry.level|junior|fresher|graduate|new.grad)\b', text_lower):
            return 'Entry Level / Junior'
        elif re.search(r'\b(?:senior|lead|principal|architect)\b', text_lower):
            return 'Senior Level'
        elif re.search(r'\b(?:mid.level|intermediate)\b', text_lower):
            return 'Mid Level'
        
        return 'Not specified'

    def _extract_education(self, text):
        """Extract education information"""
        text_lower = text.lower()
        degrees = []
        
        # Degree patterns
        degree_patterns = [
            (r'bachelor(?:\'s)?|b\.?s\.?|ba|be|btech|bsc', 'Bachelor\'s'),
            (r'master(?:\'s)?|m\.?s\.?|ma|mba|mtech|msc', 'Master\'s'),
            (r'phd|ph\.d|doctorate|doctoral', 'PhD/Doctorate'),
            (r'diploma|certificate', 'Diploma/Certificate'),
            (r'associate|aa|as', 'Associate'),
        ]
        
        for pattern, degree_type in degree_patterns:
            if re.search(pattern, text_lower):
                degrees.append(degree_type)
        
        return list(set(degrees)) if degrees else ['Not specified']

    def _extract_certifications(self, text):
        """Extract certifications"""
        text_lower = text.lower()
        certifications = []
        
        cert_patterns = [
            'aws certified',
            'microsoft certified',
            'google certified',
            'oracle certified',
            'cisco certified',
            'pmp',
            'project management professional',
            'scrum master',
            'certified scrum master',
            'itil',
            'comptia',
            'cissp',
            'ceh',
            'cisa',
            'cism'
        ]
        
        for cert in cert_patterns:
            if cert in text_lower:
                certifications.append(cert.title())
        
        return certifications if certifications else ['None mentioned']

    def _extract_companies(self, text):
        """Extract company names using NER and patterns"""
        companies = []
        
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    companies.append(ent.text)
        except:
            pass
        
        # Pattern-based extraction
        company_patterns = [
            r'worked\s+at\s+([A-Z][a-zA-Z\s&]+)',
            r'experience\s+at\s+([A-Z][a-zA-Z\s&]+)',
            r'employed\s+by\s+([A-Z][a-zA-Z\s&]+)',
            r'intern\s+at\s+([A-Z][a-zA-Z\s&]+)'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.extend([match.strip() for match in matches])
        
        # Clean and limit results
        companies = list(set([comp for comp in companies if len(comp) > 2 and len(comp) < 50]))
        return companies[:5] if companies else ['Not specified']

    def _extract_projects(self, text):
        """Extract project information"""
        text_lower = text.lower()
        
        project_count = len(re.findall(r'\bproject\b', text_lower))
        
        # Look for specific project indicators
        project_indicators = [
            'developed',
            'built',
            'created',
            'implemented',
            'designed',
            'led'
        ]
        
        project_mentions = sum(len(re.findall(rf'\\b{indicator}\\b', text_lower)) for indicator in project_indicators)
        
        if project_count > 0:
            return f"{project_count} projects mentioned"
        elif project_mentions > 3:
            return f"Multiple development activities mentioned"
        else:
            return "No projects explicitly mentioned"

    def _extract_languages(self, text):
        """Extract programming and spoken languages"""
        text_lower = text.lower()
        
        spoken_languages = ['english', 'spanish', 'french', 'german', 'chinese', 'japanese', 'arabic', 'hindi', 'portuguese', 'russian', 'italian']
        found_languages = []
        
        for lang in spoken_languages:
            if lang in text_lower or f'{lang} language' in text_lower:
                found_languages.append(lang.title())
        
        return found_languages if found_languages else ['English (assumed)']

    def _extract_contact_info(self, text):
        """Extract contact information"""
        contact = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact['email'] = email_match.group()
        
        # Phone pattern
        phone_pattern = r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact['phone'] = phone_match.group()
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact['linkedin'] = linkedin_match.group()
        
        # GitHub pattern
        github_pattern = r'github\.com/[\w-]+'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact['github'] = github_match.group()
        
        return contact if contact else {'status': 'Not found'}
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
        Comprehensive analysis of match between job and resume with NER
        
        Args:
            job_desc (str): Job description text
            resume_text (str): Resume text
            
        Returns:
            dict: Analysis results with entities and job suggestions
        """
        try:
            score = self.predict_match(job_desc, resume_text)
            job_skills = self.extract_skills(job_desc)
            resume_skills = self.extract_skills(resume_text)
            
            # Extract named entities
            entities = self.extract_named_entities(resume_text)
            
            # Get job suggestions
            job_suggestions = self.suggest_job_positions(resume_skills)
            
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
                'skills_match_ratio': round(len(matching_skills) / max(len(job_skills), 1), 2),
                'named_entities': entities,
                'job_suggestions': job_suggestions,
                'detailed_analysis': {
                    'strength_areas': self._identify_strength_areas(resume_skills),
                    'improvement_areas': missing_skills[:5],  # Top 5 missing skills
                    'career_level': self._determine_career_level(entities, resume_skills),
                    'technical_depth': len(resume_skills),
                    'domain_expertise': self._identify_domain_expertise(resume_skills)
                }
            }
            
        except Exception as e:
            return {
                'match_score': 1.0,
                'match_percentage': 20.0,
                'matching_skills': [],
                'missing_skills': [],
                'extra_skills': [],
                'error': str(e),
                'named_entities': {},
                'job_suggestions': []
            }

    def _identify_strength_areas(self, skills):
        """Identify candidate's strength areas based on skills"""
        skill_categories = {
            'Programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby'],
            'Data Science': ['machine learning', 'deep learning', 'pandas', 'numpy', 'tensorflow', 'pytorch'],
            'Web Development': ['html', 'css', 'react', 'angular', 'vue', 'nodejs', 'django', 'flask'],
            'Cloud & DevOps': ['aws', 'azure', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis'],
            'Mobile': ['android', 'ios', 'react native', 'flutter']
        }
        
        strengths = []
        skills_lower = [skill.lower() for skill in skills]
        
        for category, category_skills in skill_categories.items():
            matching_count = sum(1 for skill in category_skills if skill in skills_lower)
            if matching_count >= 2:  # At least 2 skills in the category
                strengths.append({
                    'area': category,
                    'skill_count': matching_count,
                    'percentage': round((matching_count / len(category_skills)) * 100, 1)
                })
        
        return sorted(strengths, key=lambda x: x['skill_count'], reverse=True)

    def _determine_career_level(self, entities, skills):
        """Determine career level based on entities and skills"""
        experience = entities.get('experience_level', 'Not specified')
        skill_count = len(skills)
        
        if 'senior' in experience.lower() or skill_count > 15:
            return 'Senior Level'
        elif 'entry' in experience.lower() or 'junior' in experience.lower() or skill_count < 8:
            return 'Entry Level'
        else:
            return 'Mid Level'

    def _identify_domain_expertise(self, skills):
        """Identify domain expertise based on skill patterns"""
        skills_lower = [skill.lower() for skill in skills]
        
        domains = {
            'Web Development': ['html', 'css', 'javascript', 'react', 'angular', 'vue'],
            'Data Science': ['python', 'machine learning', 'pandas', 'numpy', 'tensorflow'],
            'Cloud Computing': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'Mobile Development': ['android', 'ios', 'react native', 'flutter'],
            'DevOps': ['docker', 'kubernetes', 'jenkins', 'terraform', 'ansible']
        }
        
        domain_scores = {}
        for domain, domain_skills in domains.items():
            score = sum(1 for skill in domain_skills if skill in skills_lower)
            if score > 0:
                domain_scores[domain] = round((score / len(domain_skills)) * 100, 1)
        
        # Return the domain with highest score
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])
        # return ('General', 0) resume_text):
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