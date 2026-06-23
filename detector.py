import pandas as pd
import re
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download("stopwords")

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = []

    for w in text.split():
        if w not in stop_words:
            words.append(w)

    return " ".join(words)

df = pd.read_csv("news.csv")

df = df.dropna()

df["text"] = df["text"].apply(clean)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc*100:.2f}%")

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved!")