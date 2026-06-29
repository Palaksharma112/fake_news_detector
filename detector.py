

import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

# -------------------------
# CLEAN TEXT (IMPROVED)
# -------------------------
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return " ".join([
        w for w in text.split()
        if len(w) > 2 and w not in STOP_WORDS
    ])

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("news.csv")
df = df.dropna()

df["text"] = df["text"].apply(clean)

print(df["label"].value_counts())  # 🔥 CHECK THIS

X = df["text"]
y = df["label"]

# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# TF-IDF (FIXED)
# -------------------------
vectorizer = TfidfVectorizer(
    max_features=20000,   # reduced for stability
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# MODEL (IMPROVED)
# -------------------------
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="liblinear"
)

model.fit(X_train_vec, y_train)

# -------------------------
# EVALUATION
# -------------------------
pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# -------------------------
# SAVE
# -------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully")

print(df["label"].value_counts())