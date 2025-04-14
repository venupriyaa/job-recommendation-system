import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import pickle
import time

from utils.preprocessing import extract_text_from_pdf, preprocess_resume
from utils.model_utils import (
    load_sbert_model, 
    load_trained_models,
    recommend_jobs,
    predict_category,
    train_models,
    save_trained_models,
    save_label_encoder,
    load_label_encoder
)

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Job Recommendation System",
    page_icon="💼",
    layout="wide",
)

# Constants
MODELS_DIR = "models"
SBERT_MODEL_PATH = os.path.join(MODELS_DIR, "fine-tuned-sbert-job-skill")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")
DATA_PATH = os.path.join("data", "final_combined_jobs.csv")

# Create directory if not exists
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

# CSS for styling
st.markdown("""
<style>
    .job-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #4285F4;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .job-title {
        color: #1E3A8A;
        font-size: 20px;
        margin-bottom: 5px;
    }
    .job-category {
        color: #6B7280;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .job-match {
        color: #10B981;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 10px;
    }
    .section-title {
        color: #374151;
        font-size: 16px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 5px;
    }
    .section-content {
        color: #4B5563;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .main-title {
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 30px;
    }
    .instructions {
        background-color: #EFF6FF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-title'>AI-Powered Job Recommendation System</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This application uses advanced machine learning to match your resume 
    with suitable job opportunities. The system analyzes your skills and 
    experience to find the most relevant positions.
    """)
    
    st.header("How it Works")
    st.write("""
    1. Upload your resume in PDF format
    2. Our AI model analyzes your resume content
    3. The system predicts your most suitable job category
    4. We recommend the top 10 job matches from our database
    """)
    
    st.header("Model Information")
    st.write("""
    - Uses fine-tuned SBERT embeddings
    - Neural network matching algorithm
    - Category prediction with deep learning
    """)

# Main content
st.markdown("<div class='instructions'>", unsafe_allow_html=True)
st.markdown("### 📄 Upload Your Resume")
st.write("Please upload your resume in PDF format to get personalized job recommendations.")
st.markdown("</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a resume file", type="pdf")

# Function to load the models and initialize the system
@st.cache_resource
def initialize_system():
    # Load the data
    df = pd.read_csv(DATA_PATH)
    
    # Ensure we have a combined_text column
    if 'combined_text' not in df.columns:
        df['combined_text'] = df['job_title'] + " " + df['job_description'] + " " + df['job_skill_set']
    
    # Load SBERT model
    sbert_model = load_sbert_model(SBERT_MODEL_PATH)
    
    # Check if trained models exist
    matching_model_path = os.path.join(TRAINED_MODELS_DIR, "matching_model.h5")
    category_model_path = os.path.join(TRAINED_MODELS_DIR, "category_model.h5")
    label_encoder_path = os.path.join(TRAINED_MODELS_DIR, "label_encoder.pkl")
    
    if os.path.exists(matching_model_path) and os.path.exists(category_model_path) and os.path.exists(label_encoder_path):
        # Load existing models
        matching_model, category_model = load_trained_models(TRAINED_MODELS_DIR)
        label_encoder = load_label_encoder(TRAINED_MODELS_DIR)
        
        # Generate embeddings for all jobs
        job_embeddings = sbert_model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
        job_embeddings_df = pd.DataFrame(job_embeddings.cpu().numpy())
    else:
        # Generate embeddings for all jobs
        job_embeddings = sbert_model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
        
        # Train new models
        st.info("Training models for the first time. This may take a few minutes...")
        matching_model, category_model, label_encoder, job_embeddings_df = train_models(df, job_embeddings, sbert_model)
        
        # Save the trained models
        save_trained_models(matching_model, category_model, TRAINED_MODELS_DIR)
        save_label_encoder(label_encoder, TRAINED_MODELS_DIR)
        
        st.success("Model training complete!")
    
    return df, sbert_model, matching_model, category_model, label_encoder, job_embeddings_df

# Initialize the system
with st.spinner("Loading models and data..."):
    df, sbert_model, matching_model, category_model, label_encoder, job_embeddings_df = initialize_system()

# Process the uploaded resume
if uploaded_file is not None:
    with st.spinner("Processing your resume..."):
        # Extract text from PDF
        try:
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # Preprocess the resume text
            cleaned_resume = preprocess_resume(resume_text)
            
            # Generate embedding for the resume
            resume_embedding = sbert_model.encode([cleaned_resume], convert_to_tensor=True)
            resume_embedding_np = resume_embedding.cpu().numpy()
            
            # Predict job category
            predicted_category, category_confidence = predict_category(
                resume_embedding_np, category_model, label_encoder)
            
            # Get job recommendations
            recommendations = recommend_jobs(
                resume_embedding_np, job_embeddings_df, matching_model, df)
            
            # Display category prediction
            st.markdown("### 📊 Category Prediction")
            st.markdown(f"""
            <div style="background-color: #EFF6FF; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <p style="font-size: 18px;">Based on your resume, we predict you'd be a good fit for the 
                <span style="font-weight: bold; color: #1E3A8A;">{predicted_category}</span> category.</p>
                <p>Confidence: {category_confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display job recommendations
            st.markdown("### 💼 Top Job Recommendations")
            
            # Create columns for filtering
            col1, col2 = st.columns(2)
            
            with col1:
                # Get unique categories from recommendations
                rec_categories = list(set([rec['category'] for rec in recommendations]))
                selected_category = st.selectbox("Filter by category", 
                                             ["All Categories"] + rec_categories)
            
            with col2:
                # Sort options
                sort_option = st.selectbox("Sort by", 
                                      ["Match Score (High to Low)", 
                                       "Match Score (Low to High)"])
            
            # Filter and sort recommendations
            filtered_recommendations = recommendations
            if selected_category != "All Categories":
                filtered_recommendations = [rec for rec in recommendations 
                                          if rec['category'] == selected_category]
            
            if sort_option == "Match Score (Low to High)":
                filtered_recommendations = sorted(filtered_recommendations, 
                                               key=lambda x: x['similarity_score'])
            else:
                # Default is high to low
                filtered_recommendations = sorted(filtered_recommendations, 
                                               key=lambda x: x['similarity_score'], 
                                               reverse=True)
            
            # Display recommendations
            for i, rec in enumerate(filtered_recommendations[:10], 1):
                st.markdown(f"""
                <div class="job-card">
                    <div class="job-title">{i}. {rec['job_title']}</div>
                    <div class="job-category">Category: {rec['category']}</div>
                    <div class="job-match">Match Score: {rec['similarity_score']:.2%}</div>
                    
                    <div class="section-title">Required Skills:</div>
                    <div class="section-content">{rec['skills']}</div>
                    
                    <div class="section-title">Job Description:</div>
                    <div class="section-content">{rec['job_description'][:300]}...</div>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing the resume: {str(e)}")
            st.info("Please make sure the uploaded file is a valid PDF.")
else:
    # Display placeholders when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #6B7280;">
        <i class="fas fa-upload" style="font-size: 48px;"></i>
        <p style="font-size: 20px; margin-top: 20px;">Upload your resume to see personalized job recommendations</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 14px;">
    AI-Powered Job Recommendation System © 2025
</div>
""", unsafe_allow_html=True)