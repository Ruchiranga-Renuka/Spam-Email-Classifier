import os
from src.predict import predict_email


def test_predict_email_runs():
    # simple smoke test: returns label and score
    text = "Hello, this is a test message"
    label, score = predict_email(text)
    assert label in {"spam", "ham"}
    assert 0.0 <= score <= 1.0
