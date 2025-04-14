import re
import nltk
import fitz  # PyMuPDF
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
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
    download_nltk_resources()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = fix_spacing(text)  # Fix missing spaces
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = word_tokenize(text)  # Tokenize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    return " ".join(words)