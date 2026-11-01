"""Fine-tuned AraBERT behind the same interface as the TF-IDF baseline.

In: a DataFrame with a `text` column. Out: one probability per intent, shape (rows, 77). The model
folder carries everything raw text must pass through: the tokenizer, the weights, the label names
(in the model config) and the max length (intent_model.json), so serving cannot drift from training.
Keep this module's path stable: a registered model refers to the class by it.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_banking_intents.ml.baseline import TEXT_COLUMN

MODEL_INFO_FILE = "intent_model.json"  # written by kaggle/train_arabert.py


class AraBertClassifier:
    """A saved model folder, turned into probabilities batch by batch on the CPU."""

    def __init__(self, model_dir: Path, batch_size: int = 64):
        info = json.loads((model_dir / MODEL_INFO_FILE).read_text())
        assert info["framework"] == "transformers", f"not a transformers model folder: {model_dir}"
        self.max_length = info["max_length"]
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32)
        self.model.eval()  # once, at load: switches dropout off

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(texts[start:start + self.batch_size], truncation=True,
                                       max_length=self.max_length, padding=True, return_tensors="pt")
                chunks.append(torch.softmax(self.model(**batch).logits, dim=-1).numpy())
        return np.concatenate(chunks)


class AraBertIntentModel(mlflow.pyfunc.PythonModel):
    """The MLflow wrapper; the model folder is logged as the `model_dir` artifact."""

    def load_context(self, context) -> None:
        self.classifier = AraBertClassifier(Path(context.artifacts["model_dir"]))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.classifier.predict_proba(model_input[TEXT_COLUMN].tolist())
