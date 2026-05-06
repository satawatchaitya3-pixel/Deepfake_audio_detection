import os
import glob
import logging
import argparse
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
# Constants
# ──────────────────────────────────────────────
MODEL_PATH  = "svm_model.pkl"
SCALER_PATH = "scaler.pkl"
N_MFCC      = 13
N_FFT       = 2048
HOP_LENGTH  = 512


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
def extract_mfcc_features(audio_path: str) -> np.ndarray | None:
    """Load a WAV file and return its mean MFCC feature vector."""
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        logger.error(f"Could not load '{audio_path}': {e}")
        return None

    mfccs = librosa.feature.mfcc(
        y=audio_data, sr=sr,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return np.mean(mfccs.T, axis=0)


# ──────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────
def create_dataset(directory: str, label: int):
    """Extract MFCC features from all WAV files in a directory."""
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: '{directory}'")
        return [], []

    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    if not audio_files:
        logger.warning(f"No WAV files found in '{directory}'")
        return [], []

    X, y = [], []
    for path in audio_files:
        features = extract_mfcc_features(path)
        if features is not None:
            X.append(features)
            y.append(label)
        else:
            logger.warning(f"Skipping file: {os.path.basename(path)}")

    logger.info(f"Loaded {len(X)}/{len(audio_files)} files from '{directory}'")
    return X, y


# ──────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────
def train_model(X: np.ndarray, y: np.ndarray) -> None:
    """Train an SVM classifier and save the model and scaler to disk."""
    unique_classes = np.unique(y)

    if len(unique_classes) < 2:
        raise ValueError(
            "Training requires samples from at least 2 classes "
            f"(found: {unique_classes}). Add more audio files."
        )

    logger.info(f"Dataset — total samples: {len(X)} | classes: {unique_classes.tolist()}")

    class_counts = np.bincount(y)
    logger.info(f"Class distribution — Genuine: {class_counts[0]} | Deepfake: {class_counts[1]}")

    # Decide whether to do a train/test split
    can_split = np.min(class_counts) >= 2
    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    else:
        logger.warning(
            "Too few samples per class for a train/test split. "
            "Training on all available data — no evaluation will be shown."
        )
        X_train, y_train = X, y
        X_test, y_test = None, None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM
    logger.info("Training SVM classifier (kernel=linear)...")
    svm_classifier = SVC(kernel="linear", random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)
    logger.info("Training complete.")

    # Evaluate if test set is available
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = svm_classifier.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=["Genuine", "Deepfake"]
        )

        logger.info(f"Accuracy : {accuracy * 100:.2f}%")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(f"Classification Report:\n{report}")

    # Persist model and scaler
    joblib.dump(svm_classifier, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Model saved  → {MODEL_PATH}")
    logger.info(f"Scaler saved → {SCALER_PATH}")


# ──────────────────────────────────────────────
# Audio analysis
# ──────────────────────────────────────────────
def analyze_audio(audio_path: str) -> str:
    """Classify a single WAV file as genuine or deepfake."""
    # Validate inputs
    if not os.path.exists(audio_path):
        return "Error: File not found."
    if not audio_path.lower().endswith(".wav"):
        return "Error: Only .wav files are supported."
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return (
            "Error: Trained model not found. "
            "Please run training first (python main.py --train)."
        )

    # Load model and scaler
    try:
        svm_classifier = joblib.load(MODEL_PATH)
        scaler         = joblib.load(SCALER_PATH)
    except Exception as e:
        return f"Error loading model: {e}"

    # Extract features and predict
    features = extract_mfcc_features(audio_path)
    if features is None:
        return "Error: Could not extract features from the audio file."

    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction      = svm_classifier.predict(features_scaled)

    if prediction[0] == 0:
        result = "The input audio is classified as GENUINE."
    else:
        result = "The input audio is classified as DEEPFAKE."

    logger.info(result)
    return result


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="VoiceShield — Deepfake Voice Detection"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the SVM model using audio files in real_audio/ and deepfake_audio/",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        metavar="FILE",
        help="Path to a .wav file to classify",
    )
    parser.add_argument(
        "--genuine-dir",
        type=str,
        default="real_audio",
        help="Directory containing genuine audio files (default: real_audio/)",
    )
    parser.add_argument(
        "--deepfake-dir",
        type=str,
        default="deepfake_audio",
        help="Directory containing deepfake audio files (default: deepfake_audio/)",
    )

    args = parser.parse_args()

    # ── Training mode ──
    if args.train:
        logger.info("=== Training Mode ===")

        X_genuine,  y_genuine  = create_dataset(args.genuine_dir,  label=0)
        X_deepfake, y_deepfake = create_dataset(args.deepfake_dir, label=1)

        if not X_genuine and not X_deepfake:
            logger.error("No audio data found. Check your directory paths.")
            return

        X = np.vstack((X_genuine, X_deepfake))
        y = np.hstack((y_genuine, y_deepfake))

        try:
            train_model(X, y)
        except ValueError as e:
            logger.error(str(e))
        return

    # ── Analysis mode ──
    if args.analyze:
        logger.info("=== Analysis Mode ===")
        result = analyze_audio(args.analyze)
        print(f"\n  ▶  {result}\n")
        return

    # ── Interactive mode (no flags) ──
    logger.info("=== Interactive Mode ===")
    print("\n  VoiceShield — Deepfake Voice Detector")
    print("  ──────────────────────────────────────")
    print("  [1] Train model")
    print("  [2] Analyze a WAV file")
    print("  [3] Exit\n")

    choice = input("  Select option: ").strip()

    if choice == "1":
        genuine_dir  = input("  Genuine audio directory  [real_audio]: ").strip() or "real_audio"
        deepfake_dir = input("  Deepfake audio directory [deepfake_audio]: ").strip() or "deepfake_audio"

        X_genuine,  y_genuine  = create_dataset(genuine_dir,  label=0)
        X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)

        if not X_genuine and not X_deepfake:
            logger.error("No audio data found. Check your directory paths.")
            return

        X = np.vstack((X_genuine, X_deepfake))
        y = np.hstack((y_genuine, y_deepfake))

        try:
            train_model(X, y)
        except ValueError as e:
            logger.error(str(e))

    elif choice == "2":
        audio_path = input("  Path to WAV file: ").strip()
        result = analyze_audio(audio_path)
        print(f"\n  ▶  {result}\n")

    elif choice == "3":
        print("  Exiting.")
    else:
        print("  Invalid option. Run the script again.")


if __name__ == "__main__":
    main()