"""Fine-tuned AraBERT as an ONNX graph, behind the same interface as the other models.

In: a DataFrame with a `text` column. Out: one probability per intent, shape (rows, 77). The model
folder holds model.onnx, the tokenizer (tokenizer.json, tokenizer_config.json) and intent_model.json
(the max length). No torch and no transformers: ONNX Runtime runs the graph and the `tokenizers`
library turns text into token IDs, so an image serving this model does not need PyTorch.

MLflow copies this file into every logged ONNX model ("models from code"). Keep it self-contained:
third-party imports only.
"""

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
from tokenizers import Tokenizer

TEXT_COLUMN = "text"  # the input contract; the baseline reads the same column (training/baseline.py)
MODEL_INFO_FILE = "intent_model.json"
ONNX_FILE = "model.onnx"
THREADS_VARIABLE = "INFERENCE_THREADS"  # a serving setting; unset or 0 lets ONNX Runtime choose


class OnnxClassifier:
    """A model folder with an ONNX graph, turned into probabilities batch by batch on the CPU."""

    def __init__(self, model_dir: Path, threads: int = 0, batch_size: int = 64):
        info = json.loads((model_dir / MODEL_INFO_FILE).read_text())
        pad_token = json.loads((model_dir / "tokenizer_config.json").read_text())["pad_token"]
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(info["max_length"])
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id(pad_token), pad_token=pad_token)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # threads per request; 0 = one per core the process sees
        self.session = ort.InferenceSession(str(model_dir / ONNX_FILE), options, providers=["CPUExecutionProvider"])
        self.input_names = [graph_input.name for graph_input in self.session.get_inputs()]
        self.batch_size = batch_size

    def feed(self, texts: list[str]) -> dict[str, np.ndarray]:
        """The graph's inputs for a batch of texts, padded to the longest one."""
        encodings = self.tokenizer.encode_batch(texts)
        arrays = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        return {name: arrays[name] for name in self.input_names}

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            logits = self.session.run(None, self.feed(texts[start:start + self.batch_size]))[0].astype(np.float32)
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))  # softmax, without overflow
            chunks.append(exp / exp.sum(axis=1, keepdims=True))
        return np.concatenate(chunks)


class OnnxIntentModel(mlflow.pyfunc.PythonModel):
    """The MLflow wrapper; the model folder is logged as the `model_dir` artifact."""

    def load_context(self, context) -> None:
        threads = int(os.environ.get(THREADS_VARIABLE) or 0)
        self.classifier = OnnxClassifier(Path(context.artifacts["model_dir"]), threads=threads)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.classifier.predict_proba(model_input[TEXT_COLUMN].tolist())


mlflow.models.set_model(OnnxIntentModel())
