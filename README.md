# hate-speech-detection-tweets-ml
A machine learning model that classifies tweets as abusive, hateful, or normal, using NLP and Flask deployment.
# Detecting Hate Speech in Tweets with Advanced Machine Learning Techniques

This repository contains my B.Tech final year project.  
It detects hate / abusive tweets using NLP and machine learning and provides a simple Flask web app for real-time prediction.

---

## 🔍 Problem Statement

Online social media platforms like Twitter contain a lot of hate speech and offensive content.  
The goal of this project is to automatically classify tweets as:

- Hate / Abusive  
- Offensive  
- Normal  

so that such content can be filtered, monitored, or flagged.

---

## 🧠 Approach

1. Data Collection & Cleaning
   - Used a public hate-speech dataset from Twitter.
   - Performed preprocessing: lower-casing, removing URLs, mentions, special characters and extra spaces.

2. Text Representation
   - Used transformer models (BERT / RoBERTa) to obtain contextual embeddings for tweets.
   - Also experimented with traditional features like TF-IDF / CountVectorizer.

3. Modeling
   - Fine-tuned a transformer model for multi-class text classification.
   - Trained traditional ML models such as XGBoost for comparison.
   - Selected the best-performing model based on validation accuracy and F1-score.

4. Evaluation
   - Metrics: Accuracy, Precision, Recall and F1-Score.
   - Paid special attention to reducing false negatives (hate tweets predicted as normal).

5. Deployment with Flask
   - Built a Flask web application.
   - User enters a tweet in a text box.
   - Backend loads the trained model and returns the predicted label in real time.

---

## 🛠 Tech Stack

- Language: Python  
- NLP & ML: Transformers (BERT / RoBERTa), scikit-learn, XGBoost, Pandas, NumPy  
- Deployment: Flask  
- Tools: Jupyter Notebook, VS Code, Git & GitHub

---

## 📁 Project Structure

`text
.
├── app.py                    # Flask web app for real-time prediction
├── model.py                  # Model training / loading script
├── requirements.txt          # Python dependencies
├── HateSpeechData.csv        # Sample dataset (for demonstration)
├── templates/                # HTML templates for the web UI
│   ├── index.html
│   ├── About.html
│   └── Contact.html
└── model/                    # Saved trained model
    └── hate_speech_model.pkl
