# Validation record

Validation run in the project build environment on 2026-11-12.

## Passed

- Python syntax compilation: `python -m compileall -q backend tests`.
- Automated tests: `pytest -q` — 5 passed.
  - Dataset labeling, blank-row removal, duplicate removal, and cross-label conflict removal.
  - Model training and artifact/metrics export on a small isolated test fixture.
  - API health, metrics, prediction response, input validation, and missing-model behavior.
- Notebook structure: validated with `nbformat`; 17 cells.
- JavaScript syntax checks: `node --check frontend/src/api.js` and `node --check frontend/vite.config.js`.
- Uploaded fake data inspection: 23,481 rows; 630 rows with a blank title/body after whitespace normalization; 5,392 duplicate copies after blank-row removal; 17,459 unique fake articles after cleaning.

## Not verified in this environment

- The official University of Victoria real-news ZIP could not be downloaded because this environment could not resolve external hosts. The project includes the verified official source URL and a repeatable downloader; run it on a machine with internet access or provide `True.csv` manually.
- No production model was trained or shipped, because the complete real/fake labeled dataset was not available here. The model trainer was tested with isolated synthetic fixtures only; these fixtures are never used as application training data.
- `npm install` timed out while attempting to reach the package registry, so the Vite production build and browser-to-API flow could not be verified here. Dependencies are pinned in `frontend/package.json`; install them locally and run `npm run build`.

Do not report synthetic-fixture test results as model accuracy or as evidence of real-world fake-news detection performance.
