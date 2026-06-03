# Hate Speech Detection in Tweets

A Flask-based hate speech detection web app that classifies tweets into:
- Hate Speech
- Offensive Language
- Normal Speech

The project includes data cleaning, model training, and a web interface for live text input.

## Project Structure

- `app.py` - Flask app for serving the prediction interface.
- `train.py` - Train a logistic regression model on `dataset.csv` and save `model.pkl` and `vectorizer.pkl`.
- `clean_dataset.py` - Basic cleaning of the raw tweet dataset and export to `cleaned_dataset.csv`.
- `dataset.csv` - Original tweet dataset used for training.
- `model.pkl` - Saved trained model (generated after training).
- `vectorizer.pkl` - Saved CountVectorizer fitted to the training data (generated after training).
- `Templates/index.html` - Web UI template for entering tweets.
- `static/style.css` - Styling for the web interface.

## Requirements

- Python 3.8+
- Flask
- pandas
- numpy
- nltk
- scikit-learn

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

Windows:
```powershell
venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install flask pandas numpy nltk scikit-learn
```

4. Download NLTK stopwords (if not already installed):

```python
python -c "import nltk; nltk.download('stopwords')"
```

## Training the Model

Train the model and generate the required files:

```bash
python train.py
```

This creates `model.pkl` and `vectorizer.pkl`.

## Running the App

Start the Flask app:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Notes

- `clean_dataset.py` can be used to preprocess `dataset.csv` and save a cleaned version as `cleaned_dataset.csv`.
- If you want to retrain the model after additional cleaning or dataset changes, run `python train.py` again.

## GitHub Push Instructions

To push this project to GitHub, add a remote repository and push the `master` branch:

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin master
```
