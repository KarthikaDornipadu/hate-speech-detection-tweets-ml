import numpy as np
from flask import Flask, request, render_template
import pickle
from model import cv  # CountVectorizer should be imported if you saved it as 'cv' during training

app = Flask(_name)  # Fixed: __name_ instead of name

# Load the trained model
model = pickle.load(open('./model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', ip="", prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    input_text_transformed = cv.transform([input_text])  # Transform using CountVectorizer

    prediction = model.predict(input_text_transformed)[0]

    return render_template('index.html', ip=input_text, prediction_text=prediction)

if _name_ == "_main_":
    app.run(debug=True)
    import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import gensim
import nltk
import re
import pickle
import os

# Download stemmer
nltk.download('punkt')
stopword = gensim.parsing.preprocessing.STOPWORDS
stemmer = nltk.SnowballStemmer("english")

# Load dataset
data = pd.read_csv("./HateSpeechData.csv", encoding="ISO-8859-1")
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "Normal Speech"})
data = data[["tweet", "labels"]]

# Clean function
def clean(text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stopword])
    return text

# Apply cleaning
data["tweet"] = data["tweet"].apply(clean)

# Feature and label
x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
if not os.path.exists('model'):
    os.makedirs('model')

pickle.dump(model, open('./model/model.pkl', 'wb'))
pickle.dump(cv, open('./model/cv.pkl', 'wb'))