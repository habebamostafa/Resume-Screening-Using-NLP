import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from screener import ResumeScreener  # Import directly from screener.py

# Page configuration
st.set_page_config(
    page_title="🤖 AI Resume Screener Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'screener' not in st.session_state:
    st.session_state.screener = ResumeScreener()
    
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
    
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# Try to load model if it exists
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    
    # Check for model file
    model_path = "resume_screener_model.joblib"
    if os.path.exists(model_path):
        try:
            st.session_state.screener.load_model(model_path)
            st.session_state.model_loaded = True
        except Exception as e:
            st.warning(f"Could not load model: {str(e)}. Using advanced similarity scoring.")

# Custom CSS for premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .skill-tag {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .skill-matching {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3748;
    }
    
    .skill-missing {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #2d3748;
    }
    
    .skill-extra {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        color: #2d3748;
    }
    
    .entity-card {
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .job-suggestion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    
    .job-suggestion-card:hover {
        transform: translateY(-2px);
    }
    
    .analysis-insight {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2d3748;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 25px;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Resume Screener Pro</h1>
    <p>Advanced AI-powered resume analysis with Named Entity Recognition & Smart Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation and settings
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Analysis mode selection
    analysis_mode = st.selectbox(
        "🔍 Analysis Mode",
        ["Single Resume Analysis", "Batch Analysis", "Job Matching", "Historical Analysis"],
        help="Choose your analysis mode"
    )
    
    # Model status
    if st.session_state.model_loaded:
        st.success("✅ AI Model: Loaded")
    else:
        st.info("📊 Mode: Advanced Similarity")
    
    # Analysis settings
    st.subheader("⚙️ Settings")
    
    confidence_threshold = st.slider(
        "Match Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Minimum confidence for skill matching"
    )
    
    enable_detailed_analysis = st.checkbox(
        "Enable Detailed Analysis",
        value=True,
        help="Include domain expertise and career insights"
    )
    
    enable_ai_insights = st.checkbox(
        "Enable AI Insights",
        value=True,
        help="Generate intelligent recommendations"
    )

if analysis_mode == "Single Resume Analysis":
    # Main content layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Job Description")
        
        # Predefined job templates
        job_templates = {
            "Custom": "",
            "Data Scientist": """We are seeking a Data Scientist to join our team:
            
Required Skills:
- Python programming (pandas, numpy, scikit-learn)
- Machine learning and statistical modeling
- SQL and database management
- Data visualization (matplotlib, seaborn, plotly)
- Experience with Jupyter notebooks
- Statistical analysis and hypothesis testing
- Deep learning frameworks (TensorFlow/PyTorch)
- Big data tools (Spark, Hadoop) - preferred
            
Requirements:
- Bachelor's/Master's in Data Science, Statistics, or related field
- 2+ years of experience in data analysis
- Strong analytical and problem-solving skills
- Experience with A/B testing and experimental design""",
            
            "Full Stack Developer": """Join our development team as a Full Stack Developer:
            
Required Skills:
- Frontend: HTML5, CSS3, JavaScript, React.js
- Backend: Node.js, Express.js, Python/Django
- Databases: MongoDB, PostgreSQL, Redis
- Version control with Git/GitHub
- RESTful API development
- Cloud platforms (AWS/Azure)
- Docker containerization
- Testing frameworks (Jest, Mocha)
            
Requirements:
- Bachelor's degree in Computer Science or equivalent
- 3+ years of full-stack development experience
- Strong understanding of software architecture
- Agile/Scrum methodology experience""",
            
            "DevOps Engineer": """We need a DevOps Engineer to optimize our infrastructure:
            
Required Skills:
- Cloud platforms: AWS, Azure, or GCP
- Containerization: Docker, Kubernetes
- Infrastructure as Code: Terraform, CloudFormation
- CI/CD pipelines: Jenkins, GitLab CI, GitHub Actions
- Monitoring: Prometheus, Grafana, ELK Stack
- Linux system administration
- Scripting: Python, Bash, PowerShell
- Configuration management: Ansible, Chef
            
Requirements:
- Bachelor's degree in Engineering or related field
- 2+ years of DevOps/SRE experience
- Strong troubleshooting skills
- Experience with microservices architecture"""
        }
        
        selected_template = st.selectbox(
            "📋 Use Job Template",
            list(job_templates.keys()),
            help="Select a pre-built job description or choose Custom"
        )
        
        if selected_template != "Custom":
            job_desc = st.text_area(
                "Job Description",
                value=job_templates[selected_template],
                height=300,
                help="You can modify this template"
            )
        else:
            job_desc = st.text_area(
                "Enter job description:",
                height=300,
                placeholder="Paste the complete job description including required skills, qualifications, and responsibilities..."
            )
        
        if job_desc:
            st.caption(f"✅ Job description entered ({len(job_desc)} characters)")
    
    with col2:
        st.subheader("📋 Resume Input")
        
        upload_option = st.radio(
            "Input Method:",
            ["📁 Upload File", "📝 Paste Text"],
            horizontal=True
        )
        
        resume_text = ""
        
        if upload_option == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Upload Resume",
                type=["pdf", "docx", "txt"],
                help="Upload PDF, DOCX, or TXT files"
            )
            
            if uploaded_file:
                try:
                    with st.spinner("🔄 Processing file..."):
                        resume_text = st.session_state.screener.extract_text_from_file(uploaded_file)
                    
                    st.success(f"✅ File processed! ({len(resume_text)} characters)")
                    
                    # File info
                    file_info = {
                        "Name": uploaded_file.name,
                        "Size": f"{uploaded_file.size / 1024:.1f} KB",
                        "Type": uploaded_file.type,
                        "Words": len(resume_text.split()) if resume_text else 0
                    }
                    
                    with st.expander("📊 File Information"):
                        for key, value in file_info.items():
                            st.write(f"**{key}:** {value}")
                    
                    # Preview
                    with st.expander("👁️ Text Preview"):
                        st.text_area(
                            "Extracted content:",
                            value=resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
                            height=150,
                            disabled=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    
        else:
            resume_text = st.text_area(
                "Paste resume content:",
                height=300,
                placeholder="Paste the complete resume text here..."
            )
            
            if resume_text:
                st.caption(f"✅ Resume entered ({len(resume_text)} characters)")
    
    # Analysis section
    st.markdown("---")
    
    # Analysis button
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        if not job_desc or not resume_text:
            st.info("👆 Please provide both job description and resume to start analysis")
            analyze_disabled = True
        else:
            analyze_disabled = False
        
        if st.button(
            "🚀 Analyze with AI",
            type="primary",
            disabled=analyze_disabled,
            use_container_width=True
        ):
            # Analysis with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Text preprocessing
                status_text.text("🔄 Preprocessing texts...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                # Step 2: Skill extraction
                status_text.text("🎯 Extracting skills with NLP...")
                progress_bar.progress(40)
                time.sleep(0.5)
                
                # Step 3: Entity recognition
                status_text.text("🏷️ Performing Named Entity Recognition...")
                progress_bar.progress(60)
                time.sleep(0.5)
                
                # Step 4: AI analysis
                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(80)
                analysis = st.session_state.screener.analyze_match(job_desc, resume_text)
                
                # Step 5: Generating insights
                status_text.text("💡 Generating insights...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Store current analysis
                st.session_state.current_analysis = analysis
                
                # Add to history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'job_title': 'Custom Job',
                    'score': analysis['match_score'],
                    'analysis': analysis
                })
                
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state.current_analysis = None

# Display results if analysis is available
if st.session_state.current_analysis:
    analysis = st.session_state.current_analysis
    
    # Main results header
    st.markdown("## 📊 AI Analysis Results")
    
    # Score dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_color = "#28a745" if analysis['match_score'] >= 4 else "#ffc107" if analysis['match_score'] >= 3 else "#dc3545"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {score_color}, {score_color}aa);">
            <h2>{analysis['match_score']:.1f}/5.0</h2>
            <p>Match Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{analysis['match_percentage']:.1f}%</h2>
            <p>Compatibility</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'skills_match_ratio' in analysis:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{analysis['skills_match_ratio']:.1%}</h2>
                <p>Skills Match</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{analysis['total_resume_skills']}</h2>
            <p>Skills Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Skills Analysis", 
        "🏷️ Named Entities", 
        "💼 Job Suggestions", 
        "📈 Visual Analytics",
        "💡 AI Insights"
    ])
    
    with tab1:
        # Skills comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("✅ Matching Skills")
            if analysis['matching_skills']:
                for skill in analysis['matching_skills']:
                    st.markdown(f'<span class="skill-tag skill-matching">✅ {skill}</span>', unsafe_allow_html=True)
            else:
                st.warning("No matching skills found")
        
        with col2:
            st.subheader("❌ Missing Skills")
            if analysis['missing_skills']:
                for skill in analysis['missing_skills'][:10]:  # Limit to 10
                    st.markdown(f'<span class="skill-tag skill-missing">❌ {skill}</span>', unsafe_allow_html=True)
                if len(analysis['missing_skills']) > 10:
                    st.caption(f"... and {len(analysis['missing_skills']) - 10} more")
            else:
                st.success("🎉 All required skills present!")
        
        with col3:
            st.subheader("⭐ Additional Skills")
            if analysis['extra_skills']:
                for skill in analysis['extra_skills'][:10]:  # Limit to 10
                    st.markdown(f'<span class="skill-tag skill-extra">⭐ {skill}</span>', unsafe_allow_html=True)
                if len(analysis['extra_skills']) > 10:
                    st.caption(f"... and {len(analysis['extra_skills']) - 10} more")
            else:
                st.info("No additional skills beyond requirements")
    
    with tab2:
        st.subheader("🏷️ Named Entity Recognition Results")
        
        if 'named_entities' in analysis:
            entities = analysis['named_entities']
            
            # Display entities in cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="entity-card">
                    <h4>👨‍💼 Experience Level</h4>
                    <p>{}</p>
                </div>
                """.format(entities.get('experience_level', 'Not specified')), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="entity-card">
                    <h4>🎓 Education</h4>
                    <p>{}</p>
                </div>
                """.format(', '.join(entities.get('education', ['Not specified']))), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="entity-card">
                    <h4>📜 Certifications</h4>
                    <p>{}</p>
                </div>
                """.format(', '.join(entities.get('certifications', ['None mentioned']))), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="entity-card">
                    <h4>🏢 Companies</h4>
                    <p>{}</p>
                </div>
                """.format(', '.join(entities.get('companies', ['Not specified']))), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="entity-card">
                    <h4>🚀 Projects</h4>
                    <p>{}</p>
                </div>
                """.format(entities.get('projects', 'No projects mentioned')), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="entity-card">
                    <h4>🌍 Languages</h4>
                    <p>{}</p>
                </div>
                """.format(', '.join(entities.get('languages', ['English (assumed)']))), unsafe_allow_html=True)
            
            # Contact information if available
            if 'contact_info' in entities and entities['contact_info'].get('status') != 'Not found':
                st.subheader("📞 Contact Information")
                contact = entities['contact_info']
                contact_cols = st.columns(4)
                
                if 'email' in contact:
                    contact_cols[0].info(f"📧 {contact['email']}")
                if 'phone' in contact:
                    contact_cols[1].info(f"📱 {contact['phone']}")
                if 'linkedin' in contact:
                    contact_cols[2].info(f"💼 LinkedIn")
                if 'github' in contact:
                    contact_cols[3].info(f"🐙 GitHub")
    
    with tab3:
        st.subheader("💼 AI-Powered Job Suggestions")
        
        if 'job_suggestions' in analysis and analysis['job_suggestions']:
            for i, suggestion in enumerate(analysis['job_suggestions']):
                st.markdown(f"""
                <div class="job-suggestion-card">
                    <h4>#{i+1} {suggestion['job_title']}</h4>
                    <p><strong>Match: {suggestion['match_percentage']:.1f}%</strong></p>
                    <p>✅ Matching: {', '.join(suggestion['matching_skills'][:5])}
                    {' ...' if len(suggestion['matching_skills']) > 5 else ''}</p>
                    <p>❌ Missing: {', '.join(suggestion['missing_skills'][:3])}
                    {' ...' if len(suggestion['missing_skills']) > 3 else ''}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific job suggestions available. The candidate may have a unique skill mix.")
    
    with tab4:
        st.subheader("📈 Visual Analytics")
        
        # Skills distribution chart
        if analysis['matching_skills'] or analysis['missing_skills'] or analysis['extra_skills']:
            skills_data = {
                'Category': ['Matching', 'Missing', 'Additional'],
                'Count': [
                    len(analysis['matching_skills']),
                    len(analysis['missing_skills']),
                    len(analysis['extra_skills'])
                ],
                'Color': ['#28a745', '#dc3545', '#17a2b8']
            }
            
            fig = px.bar(
                x=skills_data['Category'],
                y=skills_data['Count'],
                color=skills_data['Category'],
                title="Skills Distribution Analysis",
                color_discrete_map={
                    'Matching': '#28a745',
                    'Missing': '#dc3545',
                    'Additional': '#17a2b8'
                }
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Score breakdown
        if 'detailed_analysis' in analysis:
            detailed = analysis['detailed_analysis']
            
            # Strength areas visualization
            if detailed.get('strength_areas'):
                st.subheader("💪 Strength Areas")
                strength_df = pd.DataFrame(detailed['strength_areas'])
                
                fig = px.bar(
                    strength_df,
                    x='area',
                    y='skill_count',
                    title="Technical Strength Areas",
                    color='percentage',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("💡 AI-Generated Insights")
        
        if enable_ai_insights and 'detailed_analysis' in analysis:
            detailed = analysis['detailed_analysis']
            
            # Career level insight
            st.markdown(f"""
            <div class="analysis-insight">
                <h4>🎯 Career Level Assessment</h4>
                <p>Based on skills and experience, this candidate appears to be at the <strong>{detailed.get('career_level', 'Unknown')}</strong> level.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Domain expertise insight
            if detailed.get('domain_expertise'):
                domain, expertise_score = detailed['domain_expertise']
                st.markdown(f"""
                <div class="analysis-insight">
                    <h4>🏆 Domain Expertise</h4>
                    <p>Primary expertise in <strong>{domain}</strong> with {expertise_score:.1f}% domain coverage.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            score = analysis['match_score']
            if score >= 4.0:
                recommendation = "🌟 **Highly Recommended**: This candidate is an excellent match with strong technical skills and relevant experience."
                color = "#28a745"
            elif score >= 3.0:
                recommendation = "✅ **Recommended**: Good candidate with most required skills. Consider for interview."
                color = "#28a745"
            elif score >= 2.0:
                recommendation = "⚠️ **Consider with Caution**: Moderate match. Evaluate if missing skills can be developed."
                color = "#ffc107"
            else:
                recommendation = "❌ **Not Recommended**: Low match for this specific position. Better suited for different roles."
                color = "#dc3545"
            
            st.markdown(f"""
            <div class="analysis-insight" style="background: linear-gradient(135deg, {color}22, {color}11); border-left: 4px solid {color};">
                <h4>🎯 Final Recommendation</h4>
                <p>{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Export functionality
    st.markdown("---")
    st.subheader("📋 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Generate detailed report
        report = f"""
AI RESUME ANALYSIS REPORT
========================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL SCORE
------------
Match Score: {analysis['match_score']:.1f}/5.0 ({analysis['match_percentage']:.1f}%)
Skills Match Ratio: {analysis.get('skills_match_ratio', 0):.1%}

SKILLS ANALYSIS
--------------
Total Skills Found: {analysis['total_resume_skills']}
Matching Skills ({len(analysis['matching_skills'])}): {', '.join(analysis['matching_skills'])}
Missing Skills ({len(analysis['missing_skills'])}): {', '.join(analysis['missing_skills'])}
Additional Skills ({len(analysis['extra_skills'])}): {', '.join(analysis['extra_skills'])}

NAMED ENTITIES
-------------
Experience Level: {analysis.get('named_entities', {}).get('experience_level', 'Not specified')}
Education: {', '.join(analysis.get('named_entities', {}).get('education', ['Not specified']))}
Certifications: {', '.join(analysis.get('named_entities', {}).get('certifications', ['None mentioned']))}
Companies: {', '.join(analysis.get('named_entities', {}).get('companies', ['Not specified']))}
Projects: {analysis.get('named_entities', {}).get('projects', 'No projects mentioned')}

JOB RECOMMENDATIONS
------------------
"""
        
        if 'job_suggestions' in analysis:
            for i, suggestion in enumerate(analysis['job_suggestions'][:3]):
                report += f"{i+1}. {suggestion['job_title']} ({suggestion['match_percentage']:.1f}% match)\n"
        
        st.download_button(
            label="📄 Download Full Report",
            data=report,
            file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        # Export to CSV for further analysis
        if st.button("📊 Export to CSV"):
            df_data = {
                'Metric': ['Match Score', 'Match Percentage', 'Skills Match Ratio', 'Total Skills', 'Matching Skills', 'Missing Skills'],
                'Value': [
                    analysis['match_score'],
                    analysis['match_percentage'],
                    analysis.get('skills_match_ratio', 0),
                    analysis['total_resume_skills'],
                    len(analysis['matching_skills']),
                    len(analysis['missing_skills'])
                ]
            }
            df = pd.DataFrame(df_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"analysis_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔄 New Analysis"):
            st.session_state.current_analysis = None
            st.rerun()

elif analysis_mode == "Historical Analysis":
    st.subheader("📈 Analysis History")
    
    if st.session_state.analysis_history:
        # Create DataFrame from history
        history_data = []
        for entry in st.session_state.analysis_history:
            history_data.append({
                'Date': entry['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Job Title': entry['job_title'],
                'Score': entry['score'],
                'Match %': entry['analysis']['match_percentage']
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualizations
        if len(history_data) > 1:
            fig = px.line(df, x='Date', y='Score', title='Score Trend Over Time')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No analysis history available. Perform some analyses to see trends here.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <h4>🤖 AI Resume Screener Pro</h4>
        <p>Powered by Advanced NLP • Named Entity Recognition • Machine Learning</p>
        <p>Built with ❤️ using Streamlit, spaCy, and Sentence Transformers</p>
    </div>
    """,
    unsafe_allow_html=True
)lib"
    if os.path.exists(model_path):
        try:
            st.session_state.screener.load_model(model_path)
            st.session_state.model_loaded = True
        except Exception as e:
            st.warning(f"Could not load model: {str(e)}. Using fallback similarity scoring.")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-skill {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem;
        display: inline-block;
    }
    .missing-skill {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem;
        display: inline-block;
    }
    .extra-skill {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("📄 Resume Screening System")
st.markdown("Upload a resume and paste a job description to evaluate the match.")

# Model status indicator
if st.session_state.model_loaded:
    st.success("✅ AI Model loaded successfully!")
else:
    st.info("ℹ️ Using similarity-based scoring (no trained model found)")

# Main content in columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Job Description")
    job_desc = st.text_area(
        "Paste job description here:",
        height=300,
        placeholder="Enter the job description including required skills, qualifications, and responsibilities..."
    )
    
    if job_desc:
        st.caption(f"✅ Job description entered ({len(job_desc)} characters)")

with col2:
    st.subheader("📋 Resume Input")
    upload_option = st.radio(
        "Choose input method:",
        ("Upload File", "Paste Text"),
        horizontal=True
    )
    
    resume_text = ""
    
    if upload_option == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt"],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_file:
            try:
                with st.spinner("Processing file..."):
                    resume_text = st.session_state.screener.extract_text_from_file(uploaded_file)
                st.success(f"✅ File processed successfully! ({len(resume_text)} characters)")
                
                # Show preview of extracted text
                with st.expander("Preview extracted text"):
                    st.text_area("Extracted content:", value=resume_text[:500] + "..." if len(resume_text) > 500 else resume_text, height=100)
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    else:
        resume_text = st.text_area(
            "Paste resume text here:",
            height=300,
            placeholder="Paste the resume content here..."
        )
        
        if resume_text:
            st.caption(f"✅ Resume text entered ({len(resume_text)} characters)")

# Analysis section
st.markdown("---")
st.subheader("🔍 Analysis")

# Analysis button
analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
with analyze_col2:
    analyze_button = st.button(
        "🚀 Analyze Match",
        type="primary",
        use_container_width=True,
        disabled=not (job_desc and resume_text)
    )

if not job_desc or not resume_text:
    st.info("👆 Please provide both job description and resume to start analysis")

elif analyze_button:
    with st.spinner("🔄 Analyzing match..."):
        try:
            analysis = st.session_state.screener.analyze_match(job_desc, resume_text)
            
            # Display results
            st.markdown("### 📊 Results")
            
            # Score visualization
            score_col1, score_col2, score_col3, score_col4 = st.columns(4)
            
            with score_col1:
                st.metric(
                    "Match Score",
                    f"{analysis['match_score']:.1f}/5.0",
                    help="Overall compatibility score"
                )
            
            with score_col2:
                st.metric(
                    "Percentage Match",
                    f"{analysis['match_percentage']:.1f}%",
                    help="Percentage representation of the match"
                )
            
            with score_col3:
                if 'skills_match_ratio' in analysis:
                    st.metric(
                        "Skills Match",
                        f"{analysis['skills_match_ratio']:.1%}",
                        help="Ratio of matching skills"
                    )
            
            with score_col4:
                if 'total_resume_skills' in analysis:
                    st.metric(
                        "Skills Found",
                        f"{analysis['total_resume_skills']}",
                        help="Total skills detected in resume"
                    )
            
            # Skills analysis
            st.markdown("### 🎯 Skills Analysis")
            
            tab1, tab2, tab3 = st.tabs([
                f"✅ Matching Skills ({len(analysis['matching_skills'])})",
                f"❌ Missing Skills ({len(analysis['missing_skills'])})",
                f"➕ Additional Skills ({len(analysis['extra_skills'])})"
            ])
            
            with tab1:
                if analysis['matching_skills']:
                    st.markdown("**These skills from the job description were found in the resume:**")
                    skills_html = ""
                    for skill in analysis['matching_skills']:
                        skills_html += f'<span class="success-skill">✅ {skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.warning("No matching skills found between job description and resume")
            
            with tab2:
                if analysis['missing_skills']:
                    st.markdown("**These required skills were not found in the resume:**")
                    skills_html = ""
                    for skill in analysis['missing_skills']:
                        skills_html += f'<span class="missing-skill">❌ {skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.success("🎉 All detectable required skills are present!")
            
            with tab3:
                if analysis['extra_skills']:
                    st.markdown("**These additional skills were found in the resume:**")
                    skills_html = ""
                    for skill in analysis['extra_skills']:
                        skills_html += f'<span class="extra-skill">⭐ {skill}</span> '
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.info("No additional skills found beyond job requirements")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            score = analysis['match_score']
            if score >= 4.0:
                st.success("🌟 **Excellent Match!** This candidate appears to be highly qualified for the position.")
            elif score >= 3.0:
                st.success("✅ **Good Match!** This candidate has most of the required qualifications.")
            elif score >= 2.0:
                st.warning("⚠️ **Moderate Match.** Consider if the candidate can learn missing skills or has equivalent experience.")
            else:
                st.error("❌ **Low Match.** This candidate may not be suitable for this specific role.")
            
            # Export results
            st.markdown("### 📋 Export Results")
            results_summary = f"""
Resume Screening Results
========================

Match Score: {analysis['match_score']:.1f}/5.0
Match Percentage: {analysis['match_percentage']:.1f}%

Matching Skills ({len(analysis['matching_skills'])}):
{', '.join(analysis['matching_skills']) if analysis['matching_skills'] else 'None'}

Missing Skills ({len(analysis['missing_skills'])}):
{', '.join(analysis['missing_skills']) if analysis['missing_skills'] else 'None'}

Additional Skills ({len(analysis['extra_skills'])}):
{', '.join(analysis['extra_skills']) if analysis['extra_skills'] else 'None'}
"""
            
            st.download_button(
                label="📄 Download Results",
                data=results_summary,
                file_name=f"resume_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            if st.checkbox("Show error details"):
                st.exception(e)

# Sidebar with instructions
st.sidebar.markdown("## 📋 Instructions")
st.sidebar.markdown("""
1. **Enter Job Description**: Paste the complete job posting in the left panel
2. **Upload Resume**: Either upload a file or paste the text
3. **Analyze**: Click the analyze button to get results
4. **Review**: Check the matching, missing, and additional skills
""")

st.sidebar.markdown("## 📁 Supported Formats")
st.sidebar.markdown("""
- **PDF** (.pdf)
- **Word** (.docx)
- **Text** (.txt)
""")

st.sidebar.markdown("## ⚙️ How It Works")
st.sidebar.markdown("""
The system uses advanced NLP to:
- Extract skills from both documents
- Calculate semantic similarity
- Provide detailed matching analysis
- Generate actionable recommendations
""")

st.sidebar.markdown("## 🔧 Technical Info")
if st.session_state.model_loaded:
    st.sidebar.success("✅ Using trained ML model")
else:
    st.sidebar.info("ℹ️ Using similarity-based analysis")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Resume Screening System | Built with Streamlit & Advanced NLP"
    "</div>",
    unsafe_allow_html=True
)