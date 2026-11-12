from pathlib import Path

import pandas as pd
import zipfile

from backend.prepare_dataset import build_dataset


def test_dataset_builder_labels_cleans_and_deduplicates(tmp_path: Path):
    fake = pd.DataFrame(
        [
            {"title": "Fake headline one", "text": "Fake article body one", "subject": "politics", "date": "2017"},
            {"title": "Fake headline one", "text": "Fake article body one", "subject": "politics", "date": "2017"},
            {"title": "Conflict headline", "text": "Shared duplicate article", "subject": "news", "date": "2017"},
            {"title": "", "text": "Body but no title", "subject": "news", "date": "2017"},
        ]
    )
    real = pd.DataFrame(
        [
            {"title": "Real headline one", "text": "Real article body one", "subject": "world", "date": "2017"},
            {"title": "Conflict headline", "text": "Shared duplicate article", "subject": "world", "date": "2017"},
        ]
    )
    fake_csv, fake_path, real_path = tmp_path / "fake.csv", tmp_path / "fake.csv.zip", tmp_path / "True.csv"
    fake.to_csv(fake_csv, index=False)
    with zipfile.ZipFile(fake_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(fake_csv, arcname="fake.csv")
    real.to_csv(real_path, index=False)

    output, report = tmp_path / "train.csv", tmp_path / "report.json"
    summary = build_dataset(fake_path, real_path, None, output, report, "https://example.test/data.zip")

    result = pd.read_csv(output)
    assert set(result.columns) == {"title", "text", "subject", "date", "label"}
    assert set(result["label"].unique()) == {0, 1}
    assert len(result) == 2
    assert result.set_index("title").loc["Fake headline one", "label"] == 1
    assert result.set_index("title").loc["Real headline one", "label"] == 0
    assert summary["removed_blank_title_or_text"] == 1
    assert summary["removed_conflicting_label_duplicates"] == 2
    assert summary["removed_duplicate_articles"] == 1
    assert report.exists()
