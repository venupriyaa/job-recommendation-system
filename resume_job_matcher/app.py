import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import time
from utils.resume_processor import process_resume
from utils.job_matcher import load_models_and_data, train_or_load_models, recommend_jobs

# Set page configuration
st.set_page_config(
    page_title="Resume Job Matcher",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        padding-top: 1rem;
        margin-top: 1rem;
    }
    .job-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1565C0;
        margin-bottom: 0.3rem;
    }
    .job-category {
        font-size: 1rem;
        color: #455A64;
        margin-bottom: 0.3rem;
    }
    .job-skills {
        font-size: 0.9rem;
        color: #546E7A;
        margin-bottom: 0.3rem;
    }
    .match-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E7D32;
    }
    .match-score-low {
        color: #C62828;
    }
    .match-score-medium {
        color: #F9A825;
    }
    .match-score-high {
        color: #2E7D32;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .category-badge {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px dashed #90A4AE;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f2f6;
        color: #78909C;
        font-size: 0.8rem;
    }
    .sidebar-content {
        padding: 1.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with app information
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/clouds/150/000000/resume.png")
    st.markdown("## About this app")
    st.markdown("This AI-powered application analyzes your resume and matches it with the most relevant job opportunities.")
    
    st.markdown("### How it works")
    st.markdown("""
    1. **Upload your resume** (PDF or DOCX)
    2. **AI processes your document** to extract and understand your skills
    3. **Get personalized job recommendations** that match your profile
    """)
    
    st.markdown("### Features")
    st.markdown("""
    - AI-powered skill extraction
    - Job category prediction
    - Personalized job recommendations
    - Match score calculation
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">Resume Job Matcher</h1>', unsafe_allow_html=True)

# Introduction with two columns
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <p style="font-size: 1.1rem; color: #455A64;">
    Find the perfect job match for your skills and experience with our AI-powered resume analyzer. 
    Simply upload your resume and let our advanced algorithms do the work.
    </p>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="text-align: center; background-color: #E3F2FD; padding: 1rem; border-radius: 10px;">
        <p style="font-weight: bold; color: #1565C0; margin-bottom: 0;">Ready to find your next career opportunity?</p>
    </div>
    """, unsafe_allow_html=True)


# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
SBERT_MODEL_PATH = os.path.join(current_dir, "models", "fine-tuned-sbert-job-skill")
JOB_DATA_PATH = os.path.join(current_dir, "data", "final_combined_jobs.csv")

# Initialize session state variables
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.sbert_model = None
    st.session_state.job_df = None
    st.session_state.job_embeddings = None
    st.session_state.matching_model = None
    st.session_state.category_model = None
    st.session_state.label_encoder = None

# Load models
@st.cache_resource
def load_all_models():
    with st.spinner("Loading models... This may take a moment."):
        # Load SBERT model and job data
        sbert_model, job_df, job_embeddings, label_encoder = load_models_and_data(
            SBERT_MODEL_PATH, JOB_DATA_PATH
        )
        
        # Train or load neural network models
        matching_model, category_model = train_or_load_models(
            job_df, job_embeddings, label_encoder
        )
        
        return sbert_model, job_df, job_embeddings, matching_model, category_model, label_encoder

# Try to load models
try:
    if not st.session_state.models_loaded:
        with st.spinner("Initializing AI models... This may take a moment."):
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate progress
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            (st.session_state.sbert_model, 
             st.session_state.job_df, 
             st.session_state.job_embeddings,
             st.session_state.matching_model,
             st.session_state.category_model,
             st.session_state.label_encoder) = load_all_models()
            st.session_state.models_loaded = True
            
            st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# File uploader with enhanced design
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📄 Upload Your Resume")
st.markdown("Supported formats: PDF, DOCX")
uploaded_file = st.file_uploader("", type=['pdf', 'docx'])
if not uploaded_file:
    st.markdown('<p style="color: #78909C; font-style: italic;">Drag and drop your resume file here or click to browse</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Process when file is uploaded
if uploaded_file is not None:
    try:
        with st.spinner("🔍 Analyzing your resume..."):
            # Add a progress bar for visual feedback
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulate processing time
                progress_bar.progress(i + 1)
            
            # Process resume
            cleaned_resume = process_resume(uploaded_file)
            
            # Get recommendations
            recommendations, predicted_category, category_confidence = recommend_jobs(
                cleaned_resume,
                st.session_state.sbert_model,
                st.session_state.job_embeddings,
                st.session_state.matching_model,
                st.session_state.category_model,
                st.session_state.job_df,
                st.session_state.label_encoder,
                top_n=10  # Show top 10 matches
            )
            
            # Display success message
            st.success("Resume analysis complete! Here are your job recommendations.")
            
            # Display predicted category with styled container
            st.markdown("""
            <div style="background-color: #E8F5E9; padding: 1rem; border-radius: 10px; margin: 1.5rem 0;">
                <h3 style="color: #2E7D32; margin-bottom: 0.5rem;">Predicted Career Category</h3>
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; font-weight: bold; color: #1B5E20;">{}</span>
                    <span style="margin-left: 1rem; background-color: #C8E6C9; padding: 0.3rem 0.6rem; border-radius: 15px; color: #2E7D32;">{:.2f}% confidence</span>
                </div>
            </div>
            """.format(predicted_category, category_confidence), unsafe_allow_html=True)
            
            # Display recommendations
            st.markdown('<h2 class="sub-header">Top Job Recommendations</h2>', unsafe_allow_html=True)
            
            # Show results in cards
            for i, rec in enumerate(recommendations, 1):
                # Determine color class based on score
                score_class = "match-score-low"
                if rec['similarity_score'] >= 0.8:
                    score_class = "match-score-high"
                elif rec['similarity_score'] >= 0.6:
                    score_class = "match-score-medium"
                
                st.markdown(f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: top;">
                        <div style="flex: 3;">
                            <p class="job-title">{i}. {rec['job_title']}</p>
                            <p class="job-category">
                                <span class="category-badge">{rec['category']}</span>
                            </p>
                            <p class="job-skills"><strong>Required Skills:</strong> {rec['skills']}</p>
                        </div>
                        <div style="flex: 1; text-align: right;">
                            <p class="match-score {score_class}">Match: {rec['similarity_score']:.4f}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error processing resume: {str(e)}")
        st.exception(e)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2025 Resume Job Matcher | AI-Powered Career Matchmaking")
st.markdown("</div>", unsafe_allow_html=True)
