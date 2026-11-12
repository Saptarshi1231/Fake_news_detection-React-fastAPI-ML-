from fastapi.testclient import TestClient

import app.main as api
import app.predictor as predictor


class FakeVectorizer:
    def transform(self, rows):
        assert len(rows) == 1
        return rows


class FakeModel:
    classes_ = [0, 1]

    def predict(self, _features):
        return [1]

    def predict_proba(self, _features):
        return [[0.2, 0.8]]


def test_health_and_prediction(monkeypatch):
    artifact = {
        "model": FakeModel(),
        "vectorizer": FakeVectorizer(),
        "stop_words": [],
        "metrics": {"accuracy": 0.9},
        "features": "title + text",
        "label_mapping": {0: "Real News", 1: "Fake News"},
    }
    monkeypatch.setattr(predictor, "load_artifact", lambda: artifact)
    monkeypatch.setattr(api, "model_status", lambda: True)
    monkeypatch.setattr(api, "get_model_metrics", lambda: {"metrics": {"accuracy": 0.9}})
    client = TestClient(api.app)

    assert client.get("/api/health").json()["model_ready"] is True
    assert client.get("/api/metrics").json()["metrics"]["accuracy"] == 0.9
    response = client.post("/api/predict", json={"title": "A sample headline", "text": "An article body"})
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["label"] == "Fake News"
    assert response.json()["probability_fake"] == 0.8


def test_blank_headline_is_rejected():
    client = TestClient(api.app)
    response = client.post("/api/predict", json={"title": "   "})
    assert response.status_code == 422


def test_missing_model_returns_service_unavailable(monkeypatch):
    def no_model(**_kwargs):
        raise predictor.ModelNotTrainedError("Train the model first.")

    monkeypatch.setattr(api, "model_status", lambda: False)
    monkeypatch.setattr(api, "predict_news", no_model)
    client = TestClient(api.app)
    assert client.get("/api/health").json()["model_ready"] is False
    response = client.post("/api/predict", json={"title": "A headline"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Train the model first."
