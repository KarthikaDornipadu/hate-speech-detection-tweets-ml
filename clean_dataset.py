import pandas as pd
import re

df = pd.read_csv("dataset.csv")

def clean(text):
    text = str(text).lower()

    text = re.sub(r'@\w+', '', text)      # remove usernames
    text = re.sub(r'http\S+', '', text)   # remove urls
    text = re.sub(r'&amp;', '', text)     # remove &amp;
    text = re.sub(r'\brt\b', '', text)    # remove RT
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # remove special chars

    text = " ".join(text.split())

    return text

df["tweet"] = df["tweet"].apply(clean)

df.to_csv("cleaned_dataset.csv", index=False)

print("cleaned_dataset.csv created successfully")