import os
import uuid
import logging
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import librosa
import numpy as np
import joblib

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# App configuration
# ──────────────────────────────────────────────
app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

MODEL_PATH  = "svm_model.pkl"
SCALER_PATH = "scaler.pkl"
UPLOAD_DIR  = "uploads"
ALLOWED_EXT = {"wav"}
N_MFCC      = 13
N_FFT       = 2048
HOP_LENGTH  = 512

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Model cache — loaded once at startup
# ──────────────────────────────────────────────
_model  = None
_scaler = None


def load_model():
    global _model, _scaler

    if _model is not None and _scaler is not None:
        return _model, _scaler

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: '{MODEL_PATH}'. Run: python main.py --train")
        return None, None

    if not os.path.exists(SCALER_PATH):
        logger.error(f"Scaler file not found: '{SCALER_PATH}'. Run: python main.py --train")
        return None, None

    try:
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Model and scaler loaded successfully.")
        return _model, _scaler
    except Exception as e:
        logger.error(f"Failed to load model/scaler: {e}")
        return None, None


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def extract_mfcc_features(audio_path: str):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        logger.error(f"Could not load audio '{audio_path}': {e}")
        return None

    mfccs = librosa.feature.mfcc(
        y=audio_data, sr=sr,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return np.mean(mfccs.T, axis=0)


def analyze_audio(audio_path: str):
    model, scaler = load_model()
    if model is None:
        return (
            "Error: Trained model not found. Run 'python main.py --train' first.",
            False,
        )

    features = extract_mfcc_features(audio_path)
    if features is None:
        return "Error: Could not extract features from the audio file.", False

    try:
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction      = model.predict(features_scaled)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return f"Error during classification: {e}", False

    is_genuine = bool(prediction[0] == 0)
    result = (
        "The input audio is classified as genuine."
        if is_genuine
        else "The input audio is classified as deepfake."
    )
    logger.info(f"Result for '{os.path.basename(audio_path)}': {result}")
    return result, is_genuine


# ──────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────
@app.errorhandler(413)
def file_too_large(e):
    return render_template(
        "index.html",
        message="File too large. Maximum allowed size is 16 MB."
    ), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return render_template(
        "index.html",
        message="An internal server error occurred. Please try again."
    ), 500


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/health")
def health():
    model, _ = load_model()
    status   = "ok" if model is not None else "model_not_found"
    code     = 200 if model is not None else 503
    return jsonify({"status": status}), code


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if "audio_file" not in request.files:
            return render_template("index.html", message="No file part in the request.")

        audio_file = request.files["audio_file"]

        if audio_file.filename == "":
            return render_template("index.html", message="No file selected.")

        if not allowed_file(audio_file.filename):
            return render_template(
                "index.html",
                message="Invalid file format. Only .wav files are accepted."
            )

        # Initialise before try block so finally never hits a NameError
        audio_path = None
        result     = "Error: Analysis could not be completed."
        is_genuine = False

        try:
            original_name = secure_filename(audio_file.filename)
            safe_filename = f"{uuid.uuid4().hex}_{original_name}"
            audio_path    = os.path.join(UPLOAD_DIR, safe_filename)

            audio_file.save(audio_path)
            logger.info(f"Uploaded: '{original_name}' -> '{safe_filename}'")

            result, is_genuine = analyze_audio(audio_path)
            logger.info(f"Analysis done. is_genuine={is_genuine}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            result     = f"An unexpected error occurred: {e}"
            is_genuine = False

        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Temp file removed: '{audio_path}'")

        return render_template("result.html", result=result, is_genuine=is_genuine)

    return render_template("index.html")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    load_model()  # pre-load at startup
    app.run(debug=True, host="0.0.0.0", port=5000)