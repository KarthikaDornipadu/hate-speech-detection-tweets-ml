# Hate Speech Detection in Tweets using Machine Learning

## Project Overview

This project is a Machine Learning and Natural Language Processing (NLP) based web application that detects whether a given tweet or text belongs to one of the following categories:

- Hate Speech
- Offensive Language
- Normal Speech

The application is developed using Python, Flask, Scikit-learn, and NLP techniques. Users can enter text through a web interface, and the model predicts the appropriate category.

---

## Problem Statement

Social media platforms contain a large amount of user-generated content. Some posts may contain hate speech, abusive language, or offensive content that can negatively impact individuals and communities.

The objective of this project is to automatically identify and classify such content using Machine Learning techniques.

---

## Dataset Used

Hate Speech and Offensive Language Dataset

Dataset Size: 24,783 Tweets

Classes:

| Class   |      Meaning       |
|---------|--------------------|
|    0    | Hate Speech        |
|    1    | Offensive Language |
|    2    | Normal Speech      |

The dataset contains tweets collected from Twitter and manually labeled into the above categories.

---

## Dataset Preprocessing

The dataset contained:

- URLs
- Special characters
- Numbers
- Punctuation marks
- Extra spaces
- Uppercase letters

To improve model performance, the tweets were cleaned using NLP preprocessing techniques.

### Cleaning Steps

1. Convert text to lowercase
2. Remove URLs
3. Remove special characters
4. Remove numbers
5. Remove punctuation
6. Remove extra spaces
7. Remove stopwords
8. Apply stemming using Porter Stemmer




## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- NLTK
- Scikit-learn
- HTML
- CSS

---

## Machine Learning Techniques

### Feature Extraction

CountVectorizer was used to convert textual data into numerical vectors.

### Classification Algorithm

Logistic Regression was used for tweet classification.

---

## Model Training Process

1. Load dataset
2. Clean tweets
3. Convert text into vectors using CountVectorizer
4. Split dataset into Training and Testing sets
5. Train Logistic Regression model
6. Evaluate model performance
7. Save trained model using Pickle

Generated Files:

model.pkl
vectorizer.pkl
---

## Model Accuracy

Overall Accuracy:

90.14%
The model was evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Classification Report

---

## Features

- Detect Hate Speech
- Detect Offensive Language
- Detect Normal Speech
- User-friendly Flask Interface
- Real-time Prediction
- NLP-based Text Processing

---

## Project Structure

HATE SPEECH DETECTION IN TWEETS
│
├── app.py
├── train.py
├── dataset.csv
├── cleaned_dataset.csv
├── model.pkl
├── vectorizer.pkl
│
├── templates
│   └── index.html
│
├── static
│   ├── style.css
│
└── README.md
---

## How to Run

Install Dependencies:

pip install flask pandas numpy nltk scikit-learn
Train Model:
python train.py

Run Flask Application:
python app.py

Open Browser:

http://127.0.0.1:5000
---

## Author

D. Karthika Chaitrika
