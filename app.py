from flask import Flask, render_template, request, session, redirect, send_from_directory, url_for
import joblib
import re
import os
import uuid
import nltk

from werkzeug.utils import secure_filename
from image_analysis import analyze_image
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
# MODEL
# ==========================================

model = None
vectorizer = None

def load_models():
    global model, vectorizer
    if model is None or vectorizer is None:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# ==========================================
# UPLOAD FOLDER
# ==========================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================================
# NLTK
# ==========================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words("english"))

# ==========================================
# CLEAN TEXT
# ==========================================

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = [
        w for w in text.split()
        if w not in STOP_WORDS
    ]

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

        db.add_user(
            request.form["username"],
            request.form["password"]
        )

        return redirect("/login")

    return render_template("register.html")

# ==========================================
# SHOW UPLOADED IMAGE
# ==========================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )

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
    image_url = None
    image_analysis = ""

    image_caption = ""
    image_verdict = ""
    image_confidence = 0
    image_objects = []

    sources = []

    total_checks = 0
    real_count = 0
    fake_count = 0
    uncertain_count = 0

    

    if request.method == "POST":

        user_input = request.form.get("news", "").strip()

        file = request.files.get("image")

        if file and file.filename != "":

            ext = os.path.splitext(file.filename)[1]

            filename = secure_filename(
                f"{uuid.uuid4().hex}{ext}"
            )

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            image_url = url_for(
                "uploaded_file",
                filename=filename
            )

            extracted_text = extract_text_from_image(filepath)

            if extracted_text.strip():
                 image_analysis = (
                    "Readable text detected in image."
                )

                 if user_input:

                    user_input += " " + extracted_text

                 else:

                    user_input = extracted_text


            else:
                 image_result = analyze_image(
                    filepath
                )

                 image_analysis = image_result.get(
                    "summary", ""
                )

                 image_caption = image_result.get(
                    "caption", ""
                )

                 image_verdict = image_result.get(
                    "verdict", ""
                )

                 image_confidence = image_result.get(
                    "confidence", 0
                )

                 image_objects = image_result.get(
                    "objects", []
                )
                    
                 if not user_input:

                    user_input = image_caption

            if not user_input:

             return render_template(
                "home.html",

                result="Please enter news or upload an image.",

                confidence=0,

                similarity=0,

                credibility=0,

                web_score=0,

                extracted_text="",

                image_analysis="",

                image_caption="",

                image_verdict="",

                image_confidence=0,

                image_objects=[],

                image_url=image_url,

                user_input="",

                sources=[],

                total_checks=total_checks,

                real_count=real_count,

                fake_count=fake_count,

                uncertain_count=uncertain_count
            )     

        # ---------------- ML ----------------

        cleaned = clean(user_input)

        model, vectorizer = load_models()
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(vec)[0]

            confidence = round(
                float(max(proba) * 100),
                2
            )

        # ---------------- WEB SEARCH ----------------

        results = search_web(user_input)

        if image_confidence > 0:

            verification = verify_sources(
                results,
                user_input,
                image_confidence
            )

        else:

            verification = verify_sources(
                results,
                user_input
            )

        similarity = verification.get("similarity", 0)
        credibility = verification.get("credibility", 0)
        web_score = verification.get("final_score", 50)
        sources = verification.get("sources", [])

        # ---------------- HYBRID SCORE ----------------

        ml_score = (
            confidence
            if str(prediction).upper() == "REAL"
            else (100 - confidence)
        )

        if image_confidence > 0:

            final_score = (

                ml_score * 0.50 +

                web_score * 0.25 +

                image_confidence * 0.25

            )

        else:

            final_score = (

                ml_score * 0.70 +

                web_score * 0.30

            )

        if final_score >= 80:
            result = "REAL NEWS "

        elif final_score >= 65:
            result = "LIKELY REAL "

        elif final_score >= 45:
            result = "UNCERTAIN ⚠️"

        else:
            result = "LIKELY FAKE ❌"

        # ---------------- SAVE HISTORY ----------------

        db.add_history(
          session["user"],
           user_input,
           result
)

    # ======================================
    # DASHBOARD
    # ======================================

    total_checks = 0
    real_count = 0
    fake_count = 0
    uncertain_count = 0

    if hasattr(db, "get_stats"):

        (
            total_checks,
            real_count,
            fake_count,
            uncertain_count
        ) = db.get_stats(session["user"])

    return render_template(
    "home.html",

    result=result,

    confidence=confidence,

    similarity=similarity,

    credibility=credibility,

    web_score=web_score,

    sources=sources,

    extracted_text=extracted_text,

    image_url=image_url,

    image_analysis=image_analysis,

    image_caption=image_caption,
        image_verdict=image_verdict,
        image_confidence=image_confidence,
        image_objects=image_objects,

    user_input=user_input,

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

    data = db.get_history(session["user"])

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
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)