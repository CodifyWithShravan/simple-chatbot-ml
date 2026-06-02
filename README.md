🤖 Simple AI Chatbot: Machine Learning Tutor

📖 Project Description
This is a foundational, rule-based Natural Language Processing (NLP) chatbot designed to act as an interactive Machine Learning tutor. The application takes raw human language queries, processes them using professional text normalization techniques, and mathematically maps them to pre-defined educational intent categories. It features a strict boundary validation threshold to handle out-of-domain conversational queries and is deployed inside a responsive, browser-based graphical web interface.

✨ Features
• Text Preprocessing: Utilizes NLTK's advanced tokenization to split strings into individual words and strips out punctuation noise.
• Text Normalization: Implements NLTK's WordNetLemmatizer to reduce words to their base dictionary root forms (e.g., converting "learning" or "learned" down to "learn"), ensuring high-accuracy keyword matching.
• Statistical Vectorization: Uses scikit-learn's TfidfVectorizer to calculate the mathematical term frequency and inverse document frequency weights across the dataset.
• Intent Classification: Computes the angular distance between vectors using cosine similarity to find the closest match index.
• Boundary Filtering Guardrails: Implements a strict mathematical validation filter threshold (0.22). Queries falling below this score are automatically flagged as out-of-domain and safely routed to a controlled fallback message handler.
• Interactive Web Interface: Built on top of the Streamlit framework with session state management to preserve chat histories during active execution.
f
🛠️ Tech Stack
• Python 3
• NLTK (Natural Language Toolkit) for tokenization and word lemmatization
• Scikit-Learn for TF-IDF matrix generation and Cosine Similarity equations
• Streamlit for the frontend chat interface UI
• JSON for the underlying intent knowledge base data structureo