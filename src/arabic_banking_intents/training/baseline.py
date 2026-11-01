"""The TF-IDF + logistic regression baseline, built as one scikit-learn Pipeline.

The vectorizer sits inside the model, so the registered model takes raw text: preprocessing travels
with the model. Its input is a DataFrame with a `text` column, the shape MLflow hands a model at
serving time, so training and serving call it the same way.
"""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TEXT_COLUMN = "text"


def build_pipeline(config: dict) -> Pipeline:
    """`config` has a `tfidf` and a `logistic_regression` table (configs/baseline.toml)."""
    tfidf = {**config["tfidf"], "ngram_range": tuple(config["tfidf"]["ngram_range"])}
    return Pipeline([
        ("features", ColumnTransformer([("tfidf", TfidfVectorizer(**tfidf), TEXT_COLUMN)])),
        ("classifier", LogisticRegression(**config["logistic_regression"])),
    ])
