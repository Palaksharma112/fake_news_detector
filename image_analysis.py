import os
import numpy as np
from PIL import Image, ImageChops, ExifTags

# ================================
# GLOBAL MODELS (LAZY LOADED)
# ================================
caption_model = None
object_model = None


# ================================
# LOAD TRANSFORMERS SAFELY
# ================================
def get_caption_model():
    global caption_model
    if caption_model is None:
        from transformers import pipeline
        caption_model = pipeline(
            task="image-to-text",
            model="Salesforce/blip-image-captioning-base"
        )
    return caption_model


def get_object_model():
    global object_model
    if object_model is None:
        from transformers import pipeline
        object_model = pipeline(
            task="object-detection",
            model="facebook/detr-resnet-50"
        )
    return object_model


# ================================
# CAPTION
# ================================
def generate_caption(image_path):
    try:
        model = get_caption_model()
        result = model(image_path)
        return result[0].get("generated_text", "")
    except:
        return "Unknown scene"


# ================================
# OBJECT DETECTION
# ================================
def detect_objects(image_path):
    try:
        detector = get_object_model()
        detections = detector(image_path)

        objects = [
            item["label"]
            for item in detections
            if item.get("score", 0) > 0.60
        ]

        return list(set(objects))
    except:
        return []


# ================================
# IMAGE HASH
# ================================
def image_hash(image_path):
    try:
        from PIL import Image
        import imagehash

        img = Image.open(image_path)
        return str(imagehash.phash(img))
    except:
        return ""


# ================================
# METADATA
# ================================
def get_metadata(image_path):
    try:
        image = Image.open(image_path)
        exif = image.getexif()

        metadata = {}
        for key, value in exif.items():
            tag = ExifTags.TAGS.get(key, key)
            metadata[tag] = value

        return metadata
    except:
        return {}


# ================================
# ELA SCORE
# ================================
def ela_score(image_path):
    try:
        original = Image.open(image_path).convert("RGB")

        temp_file = "temp_ela.jpg"
        original.save(temp_file, "JPEG", quality=90)

        compressed = Image.open(temp_file)

        diff = ImageChops.difference(original, compressed)
        extrema = diff.getextrema()

        max_diff = max([x[1] for x in extrema])

        if os.path.exists(temp_file):
            os.remove(temp_file)

        return round((max_diff / 255.0) * 100, 2)

    except:
        return 0


# ================================
# OPENCV FEATURES
# ================================
def blur_score(image):
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)


def noise_score(image):
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    noise = np.mean(np.abs(gray.astype(np.float32) - blur.astype(np.float32)))
    return round(float(noise), 2)


def edge_score(image):
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    density = np.sum(edges > 0)
    total = image.shape[0] * image.shape[1]

    return round((density / total) * 100, 2)


def brightness_score(image):
    import cv2
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return round(float(np.mean(hsv[:, :, 2])), 2)


def image_size(image):
    h, w = image.shape[:2]
    return {"width": w, "height": h}


# ================================
# MAIN ANALYSIS
# ================================
def analyze_image(image_path):

    import cv2

    image = cv2.imread(image_path)

    if image is None:
        return {
            "verdict": "INVALID IMAGE",
            "confidence": 0,
            "caption": "",
            "objects": [],
            "ela_score": 0,
            "blur_score": 0,
            "noise_score": 0,
            "edge_score": 0,
            "brightness": 0,
            "metadata": {},
            "hash": "",
            "summary": "Unable to read image."
        }

    # -----------------------------
    # AI ANALYSIS
    # -----------------------------
    caption = generate_caption(image_path)
    objects = detect_objects(image_path)
    metadata = get_metadata(image_path)

    # -----------------------------
    # FEATURES
    # -----------------------------
    ela = ela_score(image_path)
    blur = blur_score(image)
    noise = noise_score(image)
    edge = edge_score(image)
    brightness = brightness_score(image)
    size = image_size(image)
    img_hash = image_hash(image_path)

    # -----------------------------
    # SCORE CALCULATION
    # -----------------------------
    score = 50

    if blur > 350:
        score += 15
    elif blur > 180:
        score += 10
    elif blur > 80:
        score += 5
    else:
        score -= 10

    if ela < 15:
        score += 15
    elif ela < 30:
        score += 8
    elif ela < 50:
        score += 3
    else:
        score -= 10

    if 5 <= noise <= 30:
        score += 8
    elif noise > 60:
        score -= 5

    if edge > 8:
        score += 6
    elif edge < 2:
        score -= 6

    if 50 <= brightness <= 210:
        score += 6

    if size["width"] >= 1000:
        score += 5
    if size["height"] >= 1000:
        score += 5

    if metadata:
        score += 5

    if objects:
        score += 5

    score = max(0, min(100, score))

    # -----------------------------
    # VERDICT
    # -----------------------------
    if score >= 80:
        verdict = "LIKELY REAL"
    elif score >= 60:
        verdict = "UNCERTAIN"
    else:
        verdict = "LIKELY FAKE"

    # -----------------------------
    # SUMMARY
    # -----------------------------
    summary = f"""
Caption: {caption}

Objects: {', '.join(objects) if objects else 'None'}

Resolution: {size['width']} x {size['height']}

ELA Score: {ela}

Blur Score: {blur}

Noise Score: {noise}

Edge Density: {edge}

Brightness: {brightness}

Metadata Found: {'Yes' if metadata else 'No'}

Confidence: {score}%

Verdict: {verdict}
"""

    return {
        "verdict": verdict,
        "confidence": round(score, 2),
        "caption": caption,
        "objects": objects,
        "ela_score": ela,
        "blur_score": blur,
        "noise_score": noise,
        "edge_score": edge,
        "brightness": brightness,
        "metadata": metadata,
        "hash": img_hash,
        "summary": summary
    }