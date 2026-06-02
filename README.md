# 🤖 General AI Chatbot

## 📖 Project Overview
This project is a foundational, rule-based conversational agent built using Natural Language Processing (NLP) techniques. The chatbot takes raw human language input, processes it using text normalization, and mathematically maps the query to a pre-defined conversational dataset to generate the most appropriate response. 

The application is fully deployed locally using an interactive, browser-based graphical web interface.

## ✨ Features
* **Text Preprocessing:** Utilizes `NLTK`'s advanced tokenization to split strings into individual words and filter out standard punctuation.
* **Text Normalization:** Implements the `WordNetLemmatizer` to reduce words to their base dictionary root forms (e.g., converting "running" or "ran" down to "run"), ensuring high-accuracy keyword matching.
* **Statistical Vectorization:** Uses `scikit-learn`'s `TfidfVectorizer` to calculate the mathematical term frequency and inverse document frequency weights across the dataset.
* **Intent Classification:** Computes the angular distance between vectors using **Cosine Similarity** to find the closest match index.
* **Boundary Filtering Guardrails:** Implements a strict mathematical validation filter threshold. Queries that are entirely off-topic or unrecognized trigger a controlled fallback response requesting clarification.
* **Interactive Web Interface:** Built on top of the **Streamlit** framework, featuring persistent session state management to preserve chat histories during active execution.

## 🛠️ Tech Stack
* **Language:** Python 3
* **NLP Framework:** NLTK (Natural Language Toolkit)
* **Machine Learning & Math:** Scikit-Learn (TF-IDF, Cosine Similarity)
* **Frontend UI:** Streamlit
* **Data Handling:** JSON / Pandas (CSV)

## 🚀 Installation and Setup

### 1. Clone the Repository
git clone https://github.com/yourusername/general-ai-chatbot.git
cd general-ai-chatbot

### 2. Install Dependencies
Ensure you have Python installed, then run the following command to install the required libraries:
pip install streamlit nltk scikit-learn pandas

### 3. Dataset Configuration
Ensure your conversational dataset is placed in the root directory. 
* *Note: The system requires either a formatted `intents.json` file or a `dialogues.csv` file depending on the deployed backend logic.*

### 4. Run the Application
Launch the Streamlit web interface by executing the following command in your terminal:
streamlit run app.py

The application will automatically download the necessary NLTK background packages (`punkt`, `wordnet`, `omw-1.4`) on the first run and open the chat interface in your default web browser.

## 📁 Project Structure
* `app.py`: The main Python application containing the NLP pipeline and Streamlit UI code.
* `dialogues.csv` / `intents.json`: The knowledge base containing conversational patterns and responses.
* `README.md`: Project documentation.

