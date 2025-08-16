import streamlit as st
import pandas as pd
import os
from screener import ResumeScreener  # Import directly from screener.py

# Page configuration
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'screener' not in st.session_state:
    st.session_state.screener = ResumeScreener()

# Try to load model if it exists
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    
    # Check for model file
    model_path = "models/resume_screener_model.joblib"
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