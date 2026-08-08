import os
import sys

import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_model.joblib")


def predict_email(text: str):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found. Train the model first by running python src/train.py"
        )

    model = joblib.load(MODEL_PATH)
    score = model.predict_proba([text])[0][1]
    label = "spam" if score >= 0.5 else "ham"
    return label, score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"email text\"")
        sys.exit(1)

    text = sys.argv[1]
    label, score = predict_email(text)
    print(f"Prediction: {label} ({score:.4f})")
