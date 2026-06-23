from flask import Flask, render_template, request, session, redirect
import joblib
import re
import os
import nltk
from werkzeug.utils import secure_filename

from web_search import search_web
from verifier import verify_sources
from ocr_utils import extract_text_from_image
import database as db

# =========================
# APP SETUP
# =========================
app = Flask(__name__)
app.secret_key = "secret"

db.init_db()

# =========================
# MODEL LOAD
# =========================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# UPLOAD FOLDER
# =========================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# NLTK SAFE SETUP
# =========================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words("english"))

# =========================
# CLEAN TEXT
# =========================
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = [
        w for w in text.split()
        if w not in STOP_WORDS
    ]

    return " ".join(words)

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return redirect("/home" if "user" in session else "/login")


# =========================
# LOGIN
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if db.get_user(username, password):
            session["user"] = username
            return redirect("/home")

        return "Invalid Login"

    return render_template("login.html")


# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        db.add_user(
            request.form["username"],
            request.form["password"]
        )
        return redirect("/login")

    return render_template("register.html")


# =========================
# HOME (MAIN LOGIC)
# =========================
@app.route("/home", methods=["GET", "POST"])
def home():

    if "user" not in session:
        return redirect("/login")

    result = None
    confidence = 0
    similarity = 0
    credibility = 0
    web_score = 0
    sources = []

    user_input = ""
    extracted_text = ""

    if request.method == "POST":

        # =========================
        # TEXT INPUT
        # =========================
        user_input = request.form.get("news", "")

        # =========================
        # IMAGE OCR INPUT
        # =========================
        file = request.files.get("image")

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            extracted_text = extract_text_from_image(path)

            if extracted_text:
                user_input += " " + extracted_text

        if not user_input.strip():
            return render_template(
                "home.html",
                result="Please enter news or upload image."
            )

        # =========================
        # ML PREDICTION
        # =========================
        cleaned = clean(user_input)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            confidence = round(float(max(proba) * 100), 2)

        # =========================
        # WEB VERIFICATION
        # =========================
        results = search_web(user_input)
        verification = verify_sources(results, user_input)

        similarity = verification.get("similarity", 0)
        credibility = verification.get("credibility", 0)
        web_score = verification.get("final_score", 50)
        sources = verification.get("sources", [])

        # =========================
        # STABLE HYBRID SCORING
        # =========================

        ml_score = confidence if str(prediction).upper() == "REAL" else (100 - confidence)

        # clamp web score to avoid instability
        web_score = max(30, min(web_score, 90))

        score = (ml_score * 0.65) + (web_score * 0.35)

        # =========================
        # FINAL DECISION
        # =========================
        if score >= 75:
            result = "REAL NEWS ✅"
        elif score >= 60:
            result = "LIKELY REAL 🌐"
        elif score >= 45:
            result = "UNCERTAIN ⚠️"
        else:
            result = "LIKELY FAKE ❌"

        # =========================
        # SAVE HISTORY
        # =========================
        db.add_history(session["user"], user_input, result)

    # =========================
    # DASHBOARD STATS
    # =========================
    total_checks = real_count = fake_count = uncertain_count = 0

    if hasattr(db, "get_stats"):
        total_checks, real_count, fake_count, uncertain_count = db.get_stats(session["user"])

    # =========================
    # RENDER
    # =========================
    return render_template(
        "home.html",
        result=result,
        confidence=confidence,
        similarity=similarity,
        credibility=credibility,
        web_score=web_score,
        sources=sources,
        user_input=user_input,
        extracted_text=extracted_text,
        total_checks=total_checks,
        real_count=real_count,
        fake_count=fake_count,
        uncertain_count=uncertain_count
    )


# =========================
# HISTORY
# =========================
@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    data = db.get_history(session["user"])
    return render_template("history.html", data=data)


# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)