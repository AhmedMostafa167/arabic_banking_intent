"""The ONNX variants of a fine-tuned AraBERT: one function per rung of the optimisation ladder.

Each function writes a model folder (model.onnx, the tokenizer files and intent_model.json) that
training/onnx_model.py can run. The fp32 export is where every other rung starts.
"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np

INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json"]
BERT_BASE = {"num_heads": 12, "hidden_size": 768}  # the optimiser needs these to find the attention blocks


def _model_folder(source: Path, out: Path) -> Path:
    """Copy the tokenizer and model info into `out`; returns where model.onnx goes."""
    out.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        shutil.copy2(source / name, out / name)
    info = json.loads((source / "intent_model.json").read_text())
    (out / "intent_model.json").write_text(json.dumps({**info, "framework": "onnx"}))
    return out / "model.onnx"


def export_fp32(source: Path, out: Path) -> None:
    """PyTorch -> ONNX, in fp32. Batch size and sequence length stay free (dynamic axes)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    target = _model_folder(source, out)
    # "eager" attention keeps the textbook attention graph, which the optimiser knows how to fuse
    model = AutoModelForSequenceClassification.from_pretrained(source, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    sample = AutoTokenizer.from_pretrained(source)(["نص قصير", "نص تجريبي أطول قليلا"], padding=True, return_tensors="pt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the tracer's warnings about Python control flow
        torch.onnx.export(
            model, tuple(sample[name] for name in INPUT_NAMES), str(target),
            input_names=INPUT_NAMES, output_names=["logits"],
            dynamic_axes={**{name: {0: "batch", 1: "sequence"} for name in INPUT_NAMES}, "logits": {0: "batch"}},
            opset_version=17, dynamo=False,  # the classic exporter; the newer default needs onnxscript
        )


def optimise(fp32: Path, out: Path) -> dict[str, int]:
    """Fuse BERT's chains of small operations (attention, LayerNorm, GELU) into single kernels."""
    from onnxruntime.transformers.optimizer import optimize_model

    target = _model_folder(fp32, out)
    model = optimize_model(str(fp32 / "model.onnx"), model_type="bert", **BERT_BASE)
    model.save_model_to_file(str(target))
    return {name: count for name, count in model.get_fused_operator_statistics().items() if count}


def to_fp16(fp32: Path, out: Path) -> None:
    """The optimised graph with its weights and arithmetic in fp16; inputs and outputs unchanged."""
    from onnxruntime.transformers.optimizer import optimize_model

    target = _model_folder(fp32, out)
    model = optimize_model(str(fp32 / "model.onnx"), model_type="bert", **BERT_BASE)
    model.convert_float_to_float16(keep_io_types=True)
    model.save_model_to_file(str(target))


def _prepared(fp32: Path, out: Path) -> Path:
    """The fp32 graph with every tensor's type and shape written in, which quantisation needs.

    Standard ONNX shape inference only: ONNX Runtime's symbolic one gives up on our dynamic batch and
    sequence sizes, and the fused graph's custom operations hide the types quantisation looks up.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepared = out / "prepared.onnx"
    quant_pre_process(str(fp32 / "model.onnx"), str(prepared), skip_symbolic_shape=True)
    return prepared


def int8_dynamic(fp32: Path, out: Path) -> None:
    """Weights to int8 once; each activation's range is measured, and converted, on every call."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    target = _model_folder(fp32, out)
    prepared = _prepared(fp32, out)
    quantize_dynamic(str(prepared), str(target), weight_type=QuantType.QInt8)
    prepared.unlink()


def int8_selective(fp32: Path, out: Path) -> None:
    """int8 dynamic made gentler for CPUs without VNNI, where 8-bit products can saturate.

    7-bit weights (reduce_range), one scale per output channel instead of per matrix, and only the
    encoder's MatMuls quantised: the embeddings (Gather), pooler and classifier (Gemm) stay fp32.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    target = _model_folder(fp32, out)
    prepared = _prepared(fp32, out)
    quantize_dynamic(str(prepared), str(target), weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                     op_types_to_quantize=["MatMul"])
    prepared.unlink()


def int8_static(fp32: Path, out: Path, calibration: list[dict[str, np.ndarray]]) -> None:
    """Weights and activations to int8, with each activation's range fixed in advance from calibration batches."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class Batches(CalibrationDataReader):
        def __init__(self):
            self.batches = iter(calibration)

        def get_next(self):
            return next(self.batches, None)

    target = _model_folder(fp32, out)
    prepared = _prepared(fp32, out)
    quantize_static(str(prepared), str(target), Batches(), quant_format=QuantFormat.QDQ,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
    prepared.unlink()
