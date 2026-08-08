import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

from src.data_loader import load_spam_dataset


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_model.joblib")


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.9)),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
    ])


def train():
    dataset = load_spam_dataset(DATA_PATH)
    X = dataset["text"]
    y = (
        dataset["label"]
        .map(lambda v: 1 if v in {"spam", "1", "true", "yes"} else 0)
        .astype(int)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification report:\n", classification_report(y_test, predictions, target_names=["ham", "spam"]))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
