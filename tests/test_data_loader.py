import os
from src.data_loader import load_spam_dataset


def test_load_spam_dataset_exists():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
    # adjust path relative to repo root
    csv_path = os.path.abspath(csv_path)
    df = load_spam_dataset(csv_path)
    assert "text" in df.columns
    assert "label" in df.columns
    assert not df["text"].isnull().any()
