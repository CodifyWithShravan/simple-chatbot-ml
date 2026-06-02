import streamlit as st # type: ignore
import pandas as pd # type: ignore
import random
import string
import nltk # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# --- Ensure NLTK packages are downloaded silently ---
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_resources()

# Initialize the Lemmatizer
lemmatizer = WordNetLemmatizer()

# --- 1. Load the CSV Dataset ---
try:
    # Load the dataset using pandas
    df = pd.read_csv('dialogues.csv')
    
    # Drop any empty rows to prevent errors
    df = df.dropna()
    
    # Extract the columns into standard Python lists
    # Assuming your CSV has columns named 'Pattern' and 'Response'
    reference_patterns = df['Pattern'].astype(str).tolist()
    bot_responses = df['Response'].astype(str).tolist()
    
except FileNotFoundError:
    st.error("Error: 'dialogues.csv' file not found! Please place it in the same directory.")
    st.stop()
except KeyError:
    st.error("Error: The CSV must contain 'Pattern' and 'Response' columns.")
    st.stop()

# --- 2. NLTK Text Preprocessing Pipeline ---
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(cleaned_tokens)

# Preprocess all patterns from the CSV dataset
preprocessed_patterns = [preprocess_text(pat) for pat in reference_patterns]

# --- 3. Intent Matching Logic ---
def get_bot_response(user_input):
    processed_input = preprocess_text(user_input)
    
    if not processed_input.strip():
        return "Please type a valid message!"

    all_documents = preprocessed_patterns + [processed_input]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_documents)
    
    # Calculate the Cosine Similarity
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Find the best match
    best_match_index = similarity_scores.argmax()
    best_match_score = similarity_scores[0][best_match_index]
    
    # Strict boundary evaluation cutoff
    if best_match_score > 0.22:
        # Directly return the response at the matching index from the CSV
        return bot_responses[best_match_index]
    else:
        return "I'm not quite sure how to respond to that. Could you try rephrasing?"

# (The Streamlit Front-End Interface section remains exactly the same as before)