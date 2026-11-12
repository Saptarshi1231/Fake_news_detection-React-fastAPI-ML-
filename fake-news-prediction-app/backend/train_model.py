"""Train and export the Verity TF-IDF + Logistic Regression classifier.

Run from any working directory:
    python backend/train_model.py
or:
    python backend/train_model.py --data backend/data/train.csv

Label mapping is fixed throughout the project: 0 = Real, 1 = Fake.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from app.preprocessing import DEFAULT_STOP_WORDS, stemming
except ModuleNotFoundError:  # also support `python -m backend.train_model`
    from backend.app.preprocessing import DEFAULT_STOP_WORDS, stemming

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_DATA = BACKEND_DIR / "data" / "train.csv"
DEFAULT_MODEL = BACKEND_DIR / "models" / "fake_news_model.joblib"


def train(
    data_path: Path = DEFAULT_DATA,
    model_path: Path = DEFAULT_MODEL,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 100_000,
) -> dict[str, Any]:
    """Train, evaluate, save the model artifact, and write a metrics JSON file."""
    data_path = Path(data_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. First run `python backend/prepare_dataset.py`."
        )
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")

    dataset = pd.read_csv(data_path, keep_default_na=False, low_memory=False)
    dataset.columns = [str(c).strip().lower() for c in dataset.columns]
    required = {"title", "text", "label"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if dataset.empty:
        raise ValueError("Dataset is empty.")

    dataset["title"] = dataset["title"].astype(str).str.strip()
    dataset["text"] = dataset["text"].astype(str).str.strip()
    dataset = dataset[(dataset["title"] != "") & (dataset["text"] != "")].copy()
    dataset["label"] = pd.to_numeric(dataset["label"], errors="raise").astype(int)
    labels_present = set(dataset["label"].unique())
    if labels_present != {0, 1}:
        raise ValueError(
            "Training requires both classes with labels 0 (Real News) and 1 (Fake News). "
            f"Found labels: {sorted(labels_present)}"
        )
    counts = dataset["label"].value_counts()
    if int(counts.min()) < 2:
        raise ValueError("Need at least two examples in each class for stratified splitting.")

    # Split raw documents BEFORE fitting TF-IDF. This prevents vocabulary leakage.
    documents = (dataset["title"] + " " + dataset["text"]).tolist()
    labels = dataset["label"].to_numpy()
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        documents,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    stop_words = set(DEFAULT_STOP_WORDS)
    x_train = [stemming(value, stop_words) for value in x_train_raw]
    x_test = [stemming(value, stop_words) for value in x_test_raw]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1500, random_state=random_state, class_weight=None)
    model.fit(x_train_vec, y_train)
    predictions = model.predict(x_test_vec)
    probabilities_fake = model.predict_proba(x_test_vec)[:, list(model.classes_).index(1)]

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Real News", "Fake News"],
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_fake": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "roc_auc_fake": float(roc_auc_score(y_test, probabilities_fake)),
        "confusion_matrix_labels_0_real_1_fake": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
        "training_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "total_rows_used": int(len(dataset)),
        "real_rows": int((dataset["label"] == 0).sum()),
        "fake_rows": int((dataset["label"] == 1).sum()),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "test_size": float(test_size),
        "random_state": int(random_state),
    }

    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "model_version": "1.2.0",
        "trained_at_utc": trained_at,
        "model": model,
        "vectorizer": vectorizer,
        "stop_words": sorted(stop_words),
        "metrics": metrics,
        "features": "title + text",
        "label_mapping": {0: "Real News", 1: "Fake News"},
        "preprocessing": "lowercase + alphabetic cleanup + stop-word removal (negations retained) + Porter stemming",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metrics_path = model_path.with_name("metrics.json")
    metrics_payload = {
        "model_version": artifact["model_version"],
        "trained_at_utc": trained_at,
        "metrics": metrics,
        "label_mapping": {"0": "Real News", "1": "Fake News"},
        "features": artifact["features"],
        "note": "Hold-out metrics on the historical ISOT benchmark are not proof of current-news or cross-publisher generalization.",
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Verity Fake News classifier")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=100_000)
    args = parser.parse_args()

    metrics = train(
        args.data,
        args.output,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
    )
    print(f"Model saved to: {args.output.resolve()}")
    print(f"Metrics saved to: {args.output.resolve().with_name('metrics.json')}")
    print(f"Test accuracy:  {metrics['accuracy']:.4f}")
    print(f"Fake precision: {metrics['precision_fake']:.4f}")
    print(f"Fake recall:    {metrics['recall_fake']:.4f}")
    print(f"Fake F1-score:  {metrics['f1_fake']:.4f}")
    print(f"ROC-AUC:        {metrics['roc_auc_fake']:.4f}")
    print(f"Rows used:      {metrics['total_rows_used']:,}")
    print("Confusion matrix (rows=true, columns=predicted; 0=real, 1=fake):")
    for row in metrics["confusion_matrix_labels_0_real_1_fake"]:
        print(" ", row)
    print("\nCaution: random hold-out performance on ISOT does not prove generalization to current news or unseen publishers.")


if __name__ == "__main__":
    main()
