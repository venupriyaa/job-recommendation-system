# AI-Powered Job Recommendation System

This application uses deep learning to match resumes with suitable job opportunities. It analyzes resumes to predict job categories and recommend relevant positions based on skills and experience.

## Features

- Resume processing and analysis
- Job category prediction
- Personalized job recommendations
- Filtering and sorting of job recommendations
- Interactive web interface

## Tech Stack

- **Streamlit**: Web interface
- **Sentence-BERT**: Text embeddings
- **TensorFlow**: Deep learning models
- **NLTK**: Natural language processing
- **PyMuPDF**: PDF text extraction

## Project Structure

```
job_recommendation_system/
├── requirements.txt       # Project dependencies
├── app.py                 # Main Streamlit application
├── data/                  # Data files
│   └── final_combined_jobs.csv
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── model_utils.py     # Model creation and prediction functions
│   └── preprocessing.py   # Text preprocessing functions
├── models/                # Model files
│   └── fine-tuned-sbert-job-skill/
│       └── (model files)
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/job-recommendation-system.git
   cd job-recommendation-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your fine-tuned SBERT model in the `models/fine-tuned-sbert-job-skill/` directory

4. Place your job dataset in `data/final_combined_jobs.csv`

## Running the Application

To start the application, run:

```
streamlit run app.py
```

The application will be accessible in your web browser at `http://localhost:8501`.

## Usage

1. Upload your resume in PDF format
2. Wait for the system to process your resume
3. View your predicted job category
4. Explore the top 10 job recommendations
5. Filter recommendations by category or sort by match score

## First Run

On the first run, the system will train the matching and category prediction models, which may take a few minutes. These models will be saved for future use.

## Requirements

- Python 3.8+
- Internet connection (for downloading NLTK resources if not already present)
- Sufficient RAM to handle model training (minimum 4GB recommended)
- 
