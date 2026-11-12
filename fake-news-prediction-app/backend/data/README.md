# Dataset files

- `fake.csv.zip` is the fake-news file supplied by the user (23,481 rows; `title`, `text`, `subject`, `date`).
- `fake_source_inspection.json` records reproducible quality checks on that supplied file: 630 blank article bodies and 5,392 repeated title/body copies after blank rows are removed, leaving 17,459 unique fake articles.
- `True.csv` is downloaded/extracted from the matching official ISOT archive when the dataset builder runs; it is not bundled in this package.
- `train.csv` is generated locally by `python backend/prepare_dataset.py` and contains the merged binary labels.
- `dataset_info.json` records cleaning operations and actual final class counts after the two source files are merged.

Label convention: `0 = Real News`, `1 = Fake News` (kept consistent with the original notebook).

The fake-only inspection data is not a complete binary training dataset. Do not train a Fake-vs-Real model until the matching real-news file has been added.
