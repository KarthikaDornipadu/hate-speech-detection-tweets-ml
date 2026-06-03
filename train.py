import pandas as pd
import numpy as np
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Text Cleaning
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Clean tweet column
df["clean_tweet"] = df["tweet"].apply(clean)
# Input and Output
X = df["clean_tweet"]
y = df["class"]

# Convert text to numbers
cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("\nAccuracy =", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully")
print("vectorizer.pkl Saved Successfully")