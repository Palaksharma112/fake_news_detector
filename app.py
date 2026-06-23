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

# ==========================================
# APP SETUP
# ==========================================

app = Flask(__name__)
app.secret_key = "secret"

db.init_db()

# ==========================================
# MODEL LOAD
# ==========================================

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ==========================================
# OCR UPLOAD FOLDER
# ==========================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================================
# STOPWORDS
# ==========================================

nltk.download("stopwords")
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
# ==========================================
# TEXT CLEANING
# ==========================================

def clean(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z ]",
        " ",
        text
    )

    words = []

    for word in text.split():

        if word not in STOP_WORDS:

            words.append(word)

    return " ".join(words)

# ==========================================
# ROOT
# ==========================================

@app.route("/")
def index():

    if "user" in session:
        return redirect("/home")

    return redirect("/login")

# ==========================================
# LOGIN
# ==========================================

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



# ==========================================
# REGISTER
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        db.add_user(
            username,
            password
        )

        return redirect("/login")

    return render_template("register.html")

# ==========================================
# HOME
# ==========================================

@app.route("/home", methods=["GET", "POST"])
def home():

    if "user" not in session:
        return redirect("/login")

    result = None
    confidence = 0
    similarity = 0
    credibility = 0
    web_score = 0

    user_input = ""
    extracted_text = ""

    sources = []

    if request.method == "POST":

        user_input = request.form.get(
            "news",
            ""
        )

        # ==================================
        # IMAGE OCR
        # ==================================

        file = request.files.get("image")

        if file and file.filename != "":

            filename = secure_filename(
                file.filename
            )

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            extracted_text = extract_text_from_image(
                filepath
            )

            if extracted_text:

                user_input += " " + extracted_text

        # ==================================
        # NO INPUT CHECK
        # ==================================

        if not user_input.strip():

            return render_template(
                "home.html",
                result="Please enter news text or upload image."
            )

        # ==================================
        # ML PREDICTION
        # ==================================

        cleaned = clean(user_input)

        vec = vectorizer.transform(
            [cleaned]
        )

        prediction = model.predict(
            vec
        )[0]

        if hasattr(
            model,
            "predict_proba"
        ):

            proba = model.predict_proba(
                vec
            )[0]

            confidence = round(
                float(max(proba) * 100),
                2
            )

        # ==================================
        # WEB SEARCH
        # ==================================

        results = search_web(
            user_input
        )

        # ==================================
        # VERIFY
        # ==================================

        verification = verify_sources(
            results,
            user_input
        )

        similarity = verification.get(
            "similarity",
            0
        )

        credibility = verification.get(
            "credibility",
            0
        )

        web_score = verification.get(
            "final_score",
            0
        )

        sources = verification.get(
            "sources",
            []
        )

        # ==================================
        # HYBRID SCORING
        # ==================================

        score = 0

        if str(prediction).upper() == "REAL":

            score += confidence * 0.30

        else:

            score -= 10

        score += web_score * 0.70

        # ==================================
        # FINAL RESULT
        # ==================================

        if score >= 80:

            result = "REAL NEWS ✅"

        elif score >= 60:

            result = "LIKELY REAL 🌐"

        elif score >= 40:

            result = "UNCERTAIN ⚠️"

        else:

            result = "LIKELY FAKE ❌"

        # ==================================
        # SAVE HISTORY
        # ==================================

        db.add_history(
            session["user"],
            user_input,
            result
        )

    # ======================================
    # DASHBOARD STATS
    # ======================================

    total_checks = 0
    real_count = 0
    fake_count = 0
    uncertain_count = 0

    if hasattr(
        db,
        "get_stats"
    ):

        total_checks, real_count, fake_count, uncertain_count = \
            db.get_stats(
                session["user"]
            )

    # ======================================
    # RENDER PAGE
    # ======================================

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

# ==========================================
# HISTORY
# ==========================================

@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/login")

    data = db.get_history(
        session["user"]
    )

    return render_template(
        "history.html",
        data=data
    )

# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")

# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )