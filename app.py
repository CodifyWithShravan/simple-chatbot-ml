import streamlit as st # type: ignore
import json
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

# --- 1. Load the ML Knowledge Base ---
try:
    with open('intents.json', 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    st.error("Error: 'intents.json' file not found! Please place it in the same directory.")
    st.stop()

# Flatten data maps for processing
reference_patterns = []
pattern_to_intent_map = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        reference_patterns.append(pattern)
        pattern_to_intent_map.append(intent)

# --- 2. NLTK Text Preprocessing Pipeline ---
def preprocess_text(text):
    # Tokenize and convert to lowercase
    tokens = nltk.word_tokenize(text.lower())
    # Remove punctuation marks
    tokens = [token for token in tokens if token not in string.punctuation]
    # Lemmatize tokens to their base root words
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(cleaned_tokens)

preprocessed_patterns = [preprocess_text(pat) for pat in reference_patterns]

def get_bot_response(user_input):
    processed_input = preprocess_text(user_input)
    
    if not processed_input.strip():
        return "Please type a valid question about Machine Learning!"

    all_documents = preprocessed_patterns + [processed_input]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_documents)
    
    # Calculate the Cosine Similarity between the user vector (last element) and all dataset vectors
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Find the pattern that matches with highest score
    best_match_index = similarity_scores.argmax()
    best_match_score = similarity_scores[0][best_match_index]
    
    # Strict boundary evaluation cutoff to reject out-of-domain text
    if best_match_score > 0.22:
        matched_intent = pattern_to_intent_map[best_match_index]
        return random.choice(matched_intent['responses'])
    else:
        return "Sorry, I am specifically trained to answer Machine Learning questions! Try asking me about concepts like neural networks, regression, or overfitting."

# --- 4. Streamlit Front-End Interface ---
st.title("🤖 Simple AI Chatbot")
st.write("An NLP-powered Machine Learning Tutor built using NLTK and TF-IDF Vectorization.")

# Handle persistent chat interface messages array
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display entire dialogue history chain
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Collect user prompt input triggers
if prompt := st.chat_input("Ask me an ML question (e.g., 'What is supervised learning?')..."):
    # Append and render user text
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Calculate match and compute final response
    response = get_bot_response(prompt)
    
    # Append and render system response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)