import os
import pandas as pd


def load_spam_dataset(csv_path: str):
    """Load a spam dataset CSV with columns 'text' and 'label'."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin-1")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV file must contain 'text' and 'label' columns")

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(str).str.lower()
    return df
