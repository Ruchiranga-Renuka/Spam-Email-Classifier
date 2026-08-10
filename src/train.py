import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_spam_dataset
from src.preprocessing import TextPreprocessor


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_model.joblib")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def build_pipeline():
    return Pipeline([
        ("prep", TextPreprocessor()),
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

    pipeline = build_pipeline()

    param_grid = {
        "tfidf__max_df": [0.8, 0.9, 1.0],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    predictions = best.predict(X_test)
    probs = best.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary")

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Classification report:\n", classification_report(y_test, predictions, target_names=["ham", "spam"]))

    # create reports dir
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    # save model
    Path(os.path.dirname(MODEL_PATH)).mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # save metrics
    metrics_path = os.path.join(REPORTS_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write(f"Best params: {grid.best_params_}\n")
        fh.write(f"Accuracy: {acc:.4f}\n")
        fh.write(f"Precision: {precision:.4f}\n")
        fh.write(f"Recall: {recall:.4f}\n")
        fh.write(f"F1: {f1:.4f}\n")

    # confusion matrix plot
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved metrics to {metrics_path} and confusion matrix to {cm_path}")


if __name__ == "__main__":
    train()
