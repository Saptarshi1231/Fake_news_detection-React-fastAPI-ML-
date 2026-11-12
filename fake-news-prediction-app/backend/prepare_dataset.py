"""Build the project's labeled ISOT fake/real news dataset.

Default behavior:
  1. Read the user's uploaded fake.csv.zip from backend/data/.
  2. Download the official ISOT archive from the University of Victoria if
     backend/data/True.csv is not already present.
  3. Clean, label, deduplicate, shuffle, and write backend/data/train.csv.

Labels intentionally match the original project notebook:
  0 = Real News
  1 = Fake News
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
DEFAULT_FAKE = DATA_DIR / "fake.csv.zip"
DEFAULT_TRUE = DATA_DIR / "True.csv"
DEFAULT_OUTPUT = DATA_DIR / "train.csv"
DEFAULT_REPORT = DATA_DIR / "dataset_info.json"
OFFICIAL_ISOT_URL = (
    "https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/"
    "sites/7295/2023/03/News-_dataset.zip"
)
REQUIRED_COLUMNS = {"title", "text"}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FakeNewsPredictionDatasetBuilder/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if temporary.stat().st_size < 1024:
            raise RuntimeError("Downloaded archive is unexpectedly small.")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def find_csv_in_zip(archive_path: Path, target_name: str) -> str:
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            name for name in archive.namelist()
            if Path(name).name.lower() == target_name.lower()
        ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {target_name} inside {archive_path}. "
            f"Archive does not contain a member named {target_name}."
        )
    # Prefer a file at the archive root when multiple copies exist.
    return sorted(candidates, key=lambda name: (name.count("/"), len(name)))[0]


def read_csv_from_zip(archive_path: Path, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as stream:
            return pd.read_csv(stream, dtype=str, keep_default_na=False, low_memory=False)


def read_csv_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                raise ValueError(f"No CSV file found inside {path}")
            member = sorted(csv_members, key=lambda name: (name.count("/"), len(name)))[0]
            with archive.open(member) as stream:
                return pd.read_csv(stream, dtype=str, keep_default_na=False, low_memory=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def normalize_columns(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).replace("\ufeff", "").strip().lower() for column in frame.columns]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {', '.join(sorted(missing))}. "
            "Expected at least title and text."
        )
    for optional in ("subject", "date"):
        if optional not in frame.columns:
            frame[optional] = ""
    return frame[["title", "text", "subject", "date"]]


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def make_key(value: str) -> str:
    return clean_text(value).casefold()


def build_dataset(
    fake_path: Path,
    true_path: Path | None,
    source_zip: Path | None,
    output_path: Path,
    report_path: Path,
    source_url: str,
) -> dict:
    fake_frame = normalize_columns(read_csv_file(fake_path), "Fake dataset")

    # A local True.csv takes precedence. Otherwise, fetch the official source ZIP.
    if true_path is not None and true_path.exists():
        true_frame = normalize_columns(read_csv_file(true_path), "Real dataset")
        true_raw_rows = len(true_frame)
        real_source = str(true_path)
    else:
        archive_path = source_zip
        if archive_path is None:
            archive_path = DATA_DIR / "ISOT_Source_Dataset.zip"
            if not archive_path.exists():
                print("Downloading the official ISOT dataset archive from the University of Victoria...")
                try:
                    download_file(source_url, archive_path)
                except Exception as exc:
                    raise RuntimeError(
                        "Could not download the official ISOT archive. Download it manually from "
                        f"{source_url} and pass --source-zip PATH, or supply --true PATH. "
                        f"Download error: {exc}"
                    ) from exc
        true_member = find_csv_in_zip(archive_path, "True.csv")
        true_frame = normalize_columns(
            read_csv_from_zip(archive_path, true_member), "Real dataset"
        )
        true_raw_rows = len(true_frame)
        # Keep a local copy for repeatable offline runs after the first download.
        DEFAULT_TRUE.parent.mkdir(parents=True, exist_ok=True)
        true_frame.to_csv(DEFAULT_TRUE, index=False, encoding="utf-8")
        real_source = str(DEFAULT_TRUE)

    fake_raw_rows = len(fake_frame)
    fake_frame["label"] = 1  # Original notebook convention: 1 = Fake.
    true_frame["label"] = 0  # Original notebook convention: 0 = Real.
    fake_frame["_source"] = "fake"
    true_frame["_source"] = "real"
    combined = pd.concat([fake_frame, true_frame], ignore_index=True)

    for column in ("title", "text", "subject", "date"):
        combined[column] = combined[column].map(clean_text)

    blank_mask = combined["title"].eq("") | combined["text"].eq("")
    removed_blank = int(blank_mask.sum())
    combined = combined.loc[~blank_mask].copy()

    combined["_title_key"] = combined["title"].map(make_key)
    combined["_text_key"] = combined["text"].map(make_key)

    # If the identical normalized article appears with opposite labels, discard
    # that article rather than training on contradictory ground truth.
    label_counts = combined.groupby(["_title_key", "_text_key"])["label"].nunique()
    conflict_keys = label_counts[label_counts > 1].index
    conflict_mask = pd.MultiIndex.from_frame(
        combined[["_title_key", "_text_key"]]
    ).isin(conflict_keys)
    removed_conflicts = int(conflict_mask.sum())
    combined = combined.loc[~conflict_mask].copy()

    before_dedupe = len(combined)
    combined = combined.drop_duplicates(
        subset=["_title_key", "_text_key"], keep="first"
    ).copy()
    removed_duplicates = before_dedupe - len(combined)

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    result = combined[["title", "text", "subject", "date", "label"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")

    label_counts_out = result["label"].value_counts().to_dict()
    if set(label_counts_out) != {0, 1}:
        raise ValueError(
            "Cleaning must leave examples from both classes. "
            f"Remaining counts: {label_counts_out}"
        )
    report = {
        "dataset": "ISOT Fake and Real News (cleaned)",
        "label_mapping": {"0": "Real News", "1": "Fake News"},
        "source_fake": str(fake_path),
        "source_real": real_source,
        "raw_rows": {
            "fake": int(fake_raw_rows),
            "real": int(true_raw_rows),
            "total": int(fake_raw_rows + true_raw_rows),
        },
        "removed_blank_title_or_text": removed_blank,
        "removed_conflicting_label_duplicates": removed_conflicts,
        "removed_duplicate_articles": int(removed_duplicates),
        "final_rows": int(len(result)),
        "final_fake_rows": int(label_counts_out.get(1, 0)),
        "final_real_rows": int(label_counts_out.get(0, 0)),
        "columns": result.columns.tolist(),
        "shuffle_seed": 42,
        "source_url": source_url,
        "caveat": (
            "Historical English-language dataset, largely US politics/world news (2016-2017); "
            "not a universal fact-checking dataset."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nDataset created successfully")
    print(f"Output: {output_path}")
    print(f"Raw records: {report['raw_rows']['total']:,} ({fake_raw_rows:,} fake + {true_raw_rows:,} real)")
    print(f"Removed empty title/body: {removed_blank:,}")
    print(f"Removed conflicting-label copies: {removed_conflicts:,}")
    print(f"Removed duplicate articles: {removed_duplicates:,}")
    print(f"Final records: {len(result):,} ({report['final_fake_rows']:,} fake + {report['final_real_rows']:,} real)")
    print(f"Report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean, labeled Fake/Real news dataset.")
    parser.add_argument("--fake", type=Path, default=DEFAULT_FAKE, help="Fake CSV or ZIP (defaults to the bundled upload).")
    parser.add_argument("--true", type=Path, default=DEFAULT_TRUE, help="Optional local True.csv. If absent, the official source ZIP is downloaded.")
    parser.add_argument("--source-zip", type=Path, default=None, help="Use a previously downloaded official ISOT ZIP instead of downloading it.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output combined CSV path.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Output quality report JSON path.")
    parser.add_argument("--source-url", default=OFFICIAL_ISOT_URL, help="Official ISOT dataset ZIP URL.")
    args = parser.parse_args()
    true_path = args.true if args.true and args.true.exists() else None
    try:
        build_dataset(args.fake, true_path, args.source_zip, args.output, args.report, args.source_url)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
