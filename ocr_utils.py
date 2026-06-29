import cv2
import pytesseract
from PIL import Image

# Set your Tesseract path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def has_text(image_path):
    """
    Detect whether an image contains readable text.
    Returns True if text exists, otherwise False.
    """

    img = cv2.imread(image_path)

    if img is None:
        return False

    # OCR directly
    text = pytesseract.image_to_string(
        img,
        config="--oem 3 --psm 6"
    ).strip()

    # If OCR extracts enough characters, assume text exists
    return len(text) >= 10


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return gray


def extract_text_from_image(image_path):
    """
    Extract text only if readable text is present.
    """

    if not has_text(image_path):
        return ""

    img = cv2.imread(image_path)

    if img is None:
        return ""

    processed = preprocess(img)

    pil_img = Image.fromarray(processed)

    text = pytesseract.image_to_string(
        pil_img,
        config="--oem 3 --psm 6"
    )

    return text.strip()