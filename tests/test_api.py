"""The API contract, with a fake model in place of the registry: no MLflow server, no weights."""

import ast
from pathlib import Path

from fastapi.testclient import TestClient

from arabic_banking_intents.helpers.config import get_settings
from arabic_banking_intents.serving.controllers.PredictController import ServedModel
from arabic_banking_intents.serving.main import app
from arabic_banking_intents.serving.routers.dependencies import get_predictor

SERVING_DIR = Path(__file__).resolve().parents[1] / "src" / "arabic_banking_intents" / "serving"


class FakePredictor:
    info = ServedModel(name="arabic-banking-intents", alias="champion", version="7", run_id="run", model_family="fake",
                       git_commit="abc", intents=77, loaded_at="2026-11-01T00:00:00+00:00")

    def predict(self, text: str) -> list[tuple[str, float]]:
        return [("card arrival", 0.7), ("card linking", 0.2), ("exchange rate", 0.1)]


def client_with(predictor=None) -> TestClient:
    app.dependency_overrides.clear()
    if predictor is not None:
        app.dependency_overrides[get_predictor] = lambda: predictor
    return TestClient(app)  # not entered as a context manager, so the startup registry load does not run


def test_predict_returns_the_intents_and_the_version():
    response = client_with(FakePredictor()).post("/api/v1/predict", json={"text": "وين بطاقتي"})
    assert response.status_code == 200
    body = response.json()
    # The contract: an added or removed field breaks every client
    assert set(body) == {"intents", "model_version"}
    assert all(set(intent) == {"intent", "confidence"} for intent in body["intents"])
    assert [intent["intent"] for intent in body["intents"]] == ["card arrival", "card linking", "exchange rate"]
    assert body["model_version"] == "7"


def test_predict_rejects_empty_and_oversized_text():
    client = client_with(FakePredictor())
    for text in ["", "   ", "x" * (get_settings().MAX_TEXT_CHARS + 1)]:
        assert client.post("/api/v1/predict", json={"text": text}).status_code == 422


def test_not_ready_without_a_model():
    client = client_with()
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/ready").status_code == 503
    assert client.post("/api/v1/predict", json={"text": "وين بطاقتي"}).status_code == 503


def test_serving_never_imports_training():
    for path in SERVING_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not [m for m in modules if m.startswith("arabic_banking_intents.training")], f"{path} imports training code"
