from flask import Flask, render_template, request
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['tweet']

    cleaned_text = clean(text)

    vector = vectorizer.transform([cleaned_text])

    result = model.predict(vector)[0]

    if result == 0:
        prediction = "Hate Speech"

    elif result == 1:
        prediction = "Offensive Language"

    else:
        prediction = "Normal Speech"

    return render_template(
        "index.html",
        prediction=prediction,
        tweet=text
    )

if __name__ == "__main__":
    app.run(debug=True)