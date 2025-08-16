import streamlit as st
import pandas as pd
from ResumeScreener import ResumeScreener  # Assuming your class is in this module

# Initialize the screener
screener = ResumeScreener()

# Load the trained model
try:
    screener.load_model("resume_screener_model.joblib")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Streamlit UI
st.title("📄 Resume Screening System")
st.markdown("""
Upload a resume and paste a job description to evaluate the match.
""")

# Input sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    job_desc = st.text_area("Paste job description here:", height=300)

with col2:
    st.subheader("Resume Input")
    upload_option = st.radio("Choose input method:", 
                           ("Upload File", "Paste Text"))
    
    resume_text = ""
    if upload_option == "Upload File":
        uploaded_file = st.file_uploader("Choose a file", 
                                       type=["pdf", "docx", "txt"])
        if uploaded_file:
            try:
                resume_text = screener.extract_text_from_file(uploaded_file)
                st.success("File processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        resume_text = st.text_area("Paste resume text here:", height=300)

# Analysis button
if st.button("Analyze Match") and job_desc and resume_text:
    with st.spinner("Analyzing..."):
        try:
            analysis = screener.analyze_match(job_desc, resume_text)
            
            # Display results
            st.subheader("Results")
            
            # Score visualization
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Match Score", 
                         f"{analysis['match_score']:.1f}/5.0")
            with col2:
                st.metric("Percentage Match", 
                         f"{analysis['match_percentage']:.1f}%")
            
            # Skills analysis
            st.subheader("Skills Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Matching Skills", "Missing Skills", "Additional Skills"])
            
            with tab1:
                if analysis['matching_skills']:
                    st.write("These skills from the job description were found in the resume:")
                    for skill in analysis['matching_skills']:
                        st.success(f"✓ {skill}")
                else:
                    st.warning("No matching skills found")
            
            with tab2:
                if analysis['missing_skills']:
                    st.write("These required skills were not found in the resume:")
                    for skill in analysis['missing_skills']:
                        st.error(f"✗ {skill}")
                else:
                    st.success("All required skills are present!")
            
            with tab3:
                if analysis['extra_skills']:
                    st.write("These additional skills were found in the resume:")
                    for skill in analysis['extra_skills']:
                        st.info(f"☆ {skill}")
                else:
                    st.info("No additional skills found beyond job requirements")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Instructions
st.sidebar.markdown("""
### Instructions
1. Paste the job description in the left panel
2. Either upload a resume file or paste the text
3. Click "Analyze Match" button
4. View the matching results

### Supported File Formats
- PDF (.pdf)
- Word (.docx)
- Plain Text (.txt)

### How It Works
The system uses natural language processing to:
- Extract skills from both documents
- Compare semantic similarity
- Calculate a matching score
""")