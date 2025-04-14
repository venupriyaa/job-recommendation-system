import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import os

def load_sbert_model(model_path):
    """Load the fine-tuned SBERT model"""
    return SentenceTransformer(model_path)

def create_nn_matching_model(embedding_dim=384):
    """Create neural network for matching resumes to jobs"""
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
    """Create neural network for predicting job categories"""
    if num_categories is None:
        raise ValueError("num_categories must be specified")
        
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

def save_trained_models(matching_model, category_model, models_dir):
    """Save trained models to disk"""
    os.makedirs(models_dir, exist_ok=True)
    matching_model.save(os.path.join(models_dir, "matching_model.h5"))
    category_model.save(os.path.join(models_dir, "category_model.h5"))
    
def load_trained_models(models_dir):
    """Load trained models from disk"""
    matching_model = load_model(os.path.join(models_dir, "matching_model.h5"))
    category_model = load_model(os.path.join(models_dir, "category_model.h5"))
    return matching_model, category_model

def recommend_jobs(resume_embedding, job_embeddings_df, matching_model, df, top_n=10):
    """Recommend jobs based on resume embedding"""
    job_indices = job_embeddings_df.index.values
    job_embs = job_embeddings_df.values  # (N, 384)

    resume_embedding = resume_embedding.reshape(1, -1)  # Ensure shape (1, 384)
    resume_embeddings = np.repeat(resume_embedding, len(job_embs), axis=0)  # (N, 384)

    similarity_scores = matching_model.predict([resume_embeddings, job_embs], verbose=0).flatten()
    job_scores = list(zip(job_indices, similarity_scores))
    job_scores.sort(key=lambda x: x[1], reverse=True)

    top_recommendations = []
    for idx, score in job_scores[:top_n]:
        job_info = df.loc[idx]
        top_recommendations.append({
            'job_id': job_info['job_id'],
            'job_title': job_info['job_title'],
            'category': job_info['category'],
            'job_description': job_info['job_description'],
            'skills': job_info['job_skill_set'],
            'similarity_score': float(score)
        })
    return top_recommendations

def predict_category(resume_embedding, category_model, label_encoder):
    """Predict job category based on resume embedding"""
    category_probs = category_model.predict(resume_embedding.reshape(1, -1), verbose=0)[0]
    predicted_category_idx = np.argmax(category_probs)
    predicted_category = label_encoder.inverse_transform([predicted_category_idx])[0]
    category_confidence = category_probs[predicted_category_idx]
    return predicted_category, category_confidence

def train_models(df, job_embeddings, sbert_model):
    """Train matching and category prediction models"""
    # Encode categories
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    num_categories = len(label_encoder.classes_)
    
    # Convert to dataframe for easier handling
    job_embeddings_df = pd.DataFrame(job_embeddings.numpy())
    
    # Create the models
    matching_model = create_nn_matching_model()
    category_model = create_category_prediction_model(num_categories=num_categories)
    
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # Prepare data for matching model
    X_resume, X_job, y_matching = prepare_matching_data(job_embeddings_df, df)
    
    # Prepare data for category model
    X_cat = job_embeddings_df.values
    y_cat = df['category_encoded'].values
    
    # Split data
    X_resume_train, X_resume_val, X_job_train, X_job_val, y_matching_train, y_matching_val = train_test_split(
        X_resume, X_job, y_matching, test_size=0.2, random_state=42)
    
    X_cat_train, X_cat_val, y_cat_train, y_cat_val = train_test_split(
        X_cat, y_cat, test_size=0.2, stratify=y_cat, random_state=42)
    
    # Train matching model
    matching_model.fit(
        [X_resume_train, X_job_train], y_matching_train,
        validation_data=([X_resume_val, X_job_val], y_matching_val),
        epochs=5, batch_size=32, verbose=1
    )
    
    # Train category model
    category_model.fit(
        X_cat_train, y_cat_train,
        validation_data=(X_cat_val, y_cat_val),
        epochs=5, batch_size=32, verbose=1
    )
    
    return matching_model, category_model, label_encoder, job_embeddings_df

def prepare_matching_data(job_embeddings_df, df, sample_size=10000):
    """Prepare training data for matching model"""
    X_resume, X_job, y = [], [], []
    category_groups = df.groupby('category_encoded')
    categories = list(category_groups.groups.keys())
    
    pairs_created = 0
    while pairs_created < min(sample_size, len(df)):
        is_positive = np.random.choice([True, False])
        if is_positive:
            category = np.random.choice(categories)
            jobs_in_category = category_groups.get_group(category)
            if len(jobs_in_category) < 2:
                continue
            job_indices = np.random.choice(jobs_in_category.index.values, 2, replace=False)
            X_resume.append(job_embeddings_df.iloc[job_indices[0]].values)
            X_job.append(job_embeddings_df.iloc[job_indices[1]].values)
            y.append(1)
        else:
            categories_sample = np.random.choice(categories, 2, replace=False)
            job1 = np.random.choice(category_groups.get_group(categories_sample[0]).index.values)
            job2 = np.random.choice(category_groups.get_group(categories_sample[1]).index.values)
            X_resume.append(job_embeddings_df.iloc[job1].values)
            X_job.append(job_embeddings_df.iloc[job2].values)
            y.append(0)
        pairs_created += 1
    
    return np.array(X_resume), np.array(X_job), np.array(y)

def save_label_encoder(label_encoder, models_dir):
    """Save label encoder to disk"""
    import pickle
    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

def load_label_encoder(models_dir):
    """Load label encoder from disk"""
    import pickle
    with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
        return pickle.load(f)