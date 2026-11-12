from pathlib import Path

import joblib
import pandas as pd

from backend.train_model import train


def test_training_saves_predictable_artifact_and_metrics(tmp_path: Path, monkeypatch):
    rows = []
    for idx in range(40):
        rows.append({
            "title": f"Community library opens new reading room number {idx}",
            "text": f"Local education officials announced a public library reading program for residents item {idx}.",
            "label": 0,
        })
        rows.append({
            "title": f"Secret miracle cure shocks doctors headline number {idx}",
            "text": f"An unbelievable secret cure is hidden from everyone according to anonymous claims item {idx}.",
            "label": 1,
        })
    csv_path, model_path = tmp_path / "train.csv", tmp_path / "model.joblib"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    metrics = train(csv_path, model_path, max_features=2_000)
    artifact = joblib.load(model_path)

    assert model_path.exists()
    assert model_path.with_name("metrics.json").exists()
    assert set(artifact["label_mapping"].keys()) == {0, 1}
    assert "title + text" == artifact["features"]
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["training_rows"] + metrics["test_rows"] == 80
    assert len(metrics["confusion_matrix_labels_0_real_1_fake"]) == 2

    # Verify the saved training artifact can be consumed by the same predictor
    # function used by the FastAPI route.
    import app.predictor as predictor
    monkeypatch.setattr(predictor, "MODEL_PATH", model_path)
    predictor.load_artifact.cache_clear()
    prediction = predictor.predict_news(
        "Secret miracle cure shocks doctors",
        "An unbelievable secret cure is hidden from everyone according to anonymous claims.",
    )
    assert prediction["prediction"] in (0, 1)
    assert abs(prediction["probability_fake"] + prediction["probability_real"] - 1.0) < 1e-6
    predictor.load_artifact.cache_clear()
