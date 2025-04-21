import numpy as np
import pandas as pd
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import os
import pickle

def load_models_and_data(sbert_model_path, job_data_path):
    """Load the SBERT model and job data."""
    # Load SBERT model
    sbert_model = SentenceTransformer(sbert_model_path)
    
    # Load job data
    df = pd.read_csv(job_data_path)
    
    # Ensure 'combined_text' column exists
    if 'combined_text' not in df.columns:
        df['combined_text'] = df['job_title'] + " " + df['job_description'] + " " + df['job_skill_set']
    
    # Generate embeddings for all job entries
    job_embeddings = sbert_model.encode(df['combined_text'].tolist(), show_progress_bar=True, convert_to_tensor=True)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    
    return sbert_model, df, job_embeddings, label_encoder

def create_nn_matching_model(embedding_dim=384):
    """Create the neural network for job matching."""
    resume_input = Input(shape=(embedding_dim,), name="resume_embedding")
    job_input = Input(shape=(embedding_dim,), name="job_embedding")

    resume_dense = Dense(256, activation='relu')(resume_input)
    resume_dropout = Dropout(0.3)(resume_dense)
    resume_dense2 = Dense(128, activation='relu')(resume_dropout)

    job_dense = Dense(256, activation='relu')(job_input)
    job_dropout = Dropout(0.3)(job_dense)
    job_dense2 = Dense(128, activation='relu')(job_dropout)

    combined = Concatenate()([resume_dense2, job_dense2])
    dense_combined = Dense(128, activation='relu')(combined)
    dropout_combined = Dropout(0.3)(dense_combined)
    dense_combined2 = Dense(64, activation='relu')(dropout_combined)

    output = Dense(1, activation='sigmoid', name="similarity_score")(dense_combined2)

    model = Model(inputs=[resume_input, job_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_category_prediction_model(embedding_dim=384, num_categories=None):
    """Create the neural network for category prediction."""
    resume_input = Input(shape=(embedding_dim,), name="resume_embedding")
    dense1 = Dense(256, activation='relu')(resume_input)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(128, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)

    output = Dense(num_categories, activation='softmax', name="category_prediction")(dropout2)

    model = Model(inputs=resume_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_or_load_models(df, job_embeddings, label_encoder, models_dir='models'):
    """Train or load the matching and category models."""
    num_categories = len(label_encoder.classes_)
    matching_model_path = os.path.join(models_dir, 'matching_model.h5')
    category_model_path = os.path.join(models_dir, 'category_model.h5')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if we have saved models
    if os.path.exists(matching_model_path) and os.path.exists(category_model_path):
        # Load existing models
        matching_model = tf.keras.models.load_model(matching_model_path)
        category_model = tf.keras.models.load_model(category_model_path)
    else:
        # Create and train models
        print("Training new models...")
        matching_model = create_nn_matching_model()
        category_model = create_category_prediction_model(num_categories=num_categories)
        
        # Prepare training data for matching model
        X_resume, X_job, y_matching = prepare_training_data(job_embeddings.numpy(), df, sample_size=10000)
        
        # Prepare training data for category model
        X_cat, y_cat = job_embeddings.numpy(), df['category_encoded'].values
        
        # Train matching model
        matching_model.fit(
            [X_resume, X_job], y_matching,
            epochs=10, batch_size=32, verbose=1
        )
        
        # Train category model
        category_model.fit(
            X_cat, y_cat,
            epochs=10, batch_size=32, verbose=1
        )
        
        # Save models
        matching_model.save(matching_model_path)
        category_model.save(category_model_path)
    
    return matching_model, category_model

def prepare_training_data(job_embeddings, df, sample_size=10000):
    """Prepare training data for the matching model."""
    X_resume, X_job, y = [], [], []
    category_groups = df.groupby('category_encoded')
    categories = list(category_groups.groups.keys())

    pairs_created = 0
    while pairs_created < sample_size:
        is_positive = np.random.choice([True, False])
        if is_positive:
            category = np.random.choice(categories)
            jobs_in_category = category_groups.get_group(category)
            if len(jobs_in_category) < 2:
                continue
            job_indices = np.random.choice(jobs_in_category.index.values, 2, replace=False)
            X_resume.append(job_embeddings[job_indices[0]])
            X_job.append(job_embeddings[job_indices[1]])
            y.append(1)
        else:
            categories_sample = np.random.choice(categories, 2, replace=False)
            job1 = np.random.choice(category_groups.get_group(categories_sample[0]).index.values)
            job2 = np.random.choice(category_groups.get_group(categories_sample[1]).index.values)
            X_resume.append(job_embeddings[job1])
            X_job.append(job_embeddings[job2])
            y.append(0)
        pairs_created += 1
    
    return np.array(X_resume), np.array(X_job), np.array(y)

def recommend_jobs(resume_text, sbert_model, job_embeddings, matching_model, category_model, 
                   df, label_encoder, top_n=10):
    """Generate job recommendations based on resume text."""
    # Generate embedding for resume
    resume_embedding = sbert_model.encode([resume_text], convert_to_tensor=True)
    resume_embedding_np = resume_embedding.cpu().numpy()
    
    # Predict job category
    category_probs = category_model.predict(resume_embedding_np, verbose=0)[0]
    predicted_category_idx = np.argmax(category_probs)
    predicted_category = label_encoder.inverse_transform([predicted_category_idx])[0]
    category_confidence = float(category_probs[predicted_category_idx] * 100)  # Convert to percentage
    
    # Create DataFrame for job embeddings
    job_embeddings_np = job_embeddings.cpu().numpy()
    
    # Prepare for similarity prediction
    job_indices = np.arange(len(job_embeddings_np))
    resume_embeddings = np.repeat(resume_embedding_np, len(job_embeddings_np), axis=0)
    
    # Calculate similarity scores
    similarity_scores = matching_model.predict([resume_embeddings, job_embeddings_np], verbose=0).flatten()
    
    # Sort jobs by similarity score
    job_scores = list(zip(job_indices, similarity_scores))
    job_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    top_recommendations = []
    for idx, score in job_scores[:top_n]:
        job_info = df.iloc[idx]
        job_id = job_info['job_id'] if 'job_id' in job_info else f"[{idx}]"
        title = job_info['job_title']
        if isinstance(job_id, str) and job_id.strip():
            if not job_id.startswith('['):
                title = f"{title} [{job_id}]"
        
        top_recommendations.append({
            'job_title': title,
            'category': job_info['category'],
            'skills': job_info['job_skill_set'],
            'similarity_score': float(score)
        })
    
    return top_recommendations, predicted_category, category_confidence