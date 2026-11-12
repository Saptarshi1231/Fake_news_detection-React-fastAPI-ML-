# Model Card — Verity ISOT Text Classifier

## Summary

Verity is a classroom/demo text classifier that predicts whether a news article resembles the fake or real examples in the historical ISOT training corpus. It uses TF-IDF word unigrams/bigrams and Logistic Regression.

## Intended use

- Educational demonstration of data preparation, NLP preprocessing, model evaluation, REST APIs, and a React frontend.
- A preliminary screening signal to prompt further source checking.

## Out-of-scope use

- Definitively declaring that a current article is true or false.
- Election, medical, legal, financial, safety, or other high-impact decisions.
- Moderation or removal decisions without human review.
- Claims about languages or publishers absent from the training data.

## Training data

The project uses the ISOT Fake News dataset published by the University of Victoria ISOT Research Lab. It contains 23,481 fake-news articles and 21,417 real-news articles before cleaning. The provided fake file is included; the real-news file is downloaded from the official archive by the dataset builder.

The corpus is English-language, mostly focused on US politics and world news, and largely dates from 2016–2017. The dataset authors report that truthful articles were crawled from Reuters and fake articles came from sources flagged as unreliable by PolitiFact and Wikipedia. This creates a serious source/style confound: the model may learn outlet-specific writing patterns, Reuters datelines, or topic differences rather than factuality.

## Labels

- `0`: Real News
- `1`: Fake News

These are the dataset labels, not ground truth independently re-verified by this application.

## Model and preprocessing

- Concatenate `title + text`.
- Lowercase and remove non-letter characters.
- Remove a bundled English stop-word set while retaining `no`, `nor`, and `not`.
- Apply Porter stemming.
- Fit a TF-IDF word vectorizer with unigrams/bigrams on the training partition only.
- Fit Logistic Regression.
- Use a stratified random 80/20 hold-out split with a fixed seed.

## Evaluation

After training, the application stores accuracy, fake-class precision/recall/F1, ROC-AUC, per-class classification metrics, and a confusion matrix in `backend/models/metrics.json`. Metrics are not available until the complete two-class dataset has been prepared and the model has been trained. No real-data benchmark result is claimed by this project package.

## Limitations and risks

- Historical dataset; temporal drift is expected.
- Source and publisher cues can leak the label.
- Random hold-out split does not test generalization to unseen outlets or current events.
- Dataset labels reflect the original collectors' source-based labeling process.
- Logistic Regression outputs are not automatically calibrated probabilities.
- Headlines can be misleading without full context; article body is recommended but optional.

## Recommendations for stronger research evaluation

1. Add source/publisher-grouped test data from multiple independent outlets.
2. Evaluate on a later time period and a different dataset.
3. Inspect false positives and false negatives manually.
4. Compare headline-only against title-plus-body models.
5. Calibrate probabilities on a held-out calibration set before interpreting scores as confidence.
