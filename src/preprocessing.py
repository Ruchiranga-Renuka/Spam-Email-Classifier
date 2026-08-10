import re
from typing import List


class TextPreprocessor:
    """Simple text preprocessing transformer with scikit-learn style API.

    Usage: include as a step in an sklearn Pipeline.
    """

    def __init__(self):
        # compile regexes
        self.url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        self.email_re = re.compile(r"\S+@\S+", re.IGNORECASE)
        self.num_re = re.compile(r"\b\d+(?:[\,\.]\d+)*\b")
        self.nonword_re = re.compile(r"[^\w\s]")

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> List[str]:
        return [self._clean_text(t) for t in X]

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = self.url_re.sub(" URL ", text)
        text = self.email_re.sub(" EMAIL ", text)
        text = self.num_re.sub(" NUMBER ", text)
        text = self.nonword_re.sub(" ", text)
        text = " ".join(text.split())
        return text

    # scikit-learn compatibility
    def fit_transform(self, X, y=None):
        return self.transform(X)
