import os
import sys
import argparse
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_model.joblib")


def batch_predict(input_csv: str, output_csv: str):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train the model first by running python src/train.py")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_csv, encoding="latin-1")

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    texts = df["text"].fillna("")
    probs = model.predict_proba(texts)[:, 1]
    labels = ["spam" if p >= 0.5 else "ham" for p in probs]

    out = df.copy()
    out["predicted_label"] = labels
    out["spam_score"] = probs
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Wrote predictions to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Batch predict spam labels from a CSV with a 'text' column")
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    args = parser.parse_args()
    batch_predict(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
