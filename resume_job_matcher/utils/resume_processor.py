import fitz
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import docx2txt

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file upload."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + " "  # Extract text from each page
    return text.strip()

def fix_spacing(text):
    """Fix missing spaces between words using regex."""
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between camel case words
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)  # Add space between words and numbers
    text = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', text)  # Add space between numbers and words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase & uppercase transitions
    return text

def preprocess_resume(text):
    """Preprocess resume text: Fix spacing, remove unwanted characters, lemmatize, and remove stopwords."""
    text = fix_spacing(text)  # Fix missing spaces
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = word_tokenize(text)  # Tokenize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    return " ".join(words)

def process_resume(uploaded_file):
    """Process the uploaded resume file and return cleaned text."""
    if uploaded_file.name.endswith('.pdf'):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        resume_text = docx2txt.process(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")
    
    cleaned_resume = preprocess_resume(resume_text)
    return cleaned_resume