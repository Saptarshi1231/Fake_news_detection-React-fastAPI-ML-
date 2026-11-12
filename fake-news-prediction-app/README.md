# Verity — Fake News Prediction (React + FastAPI + Machine Learning)

Verity is a full-stack news-text screening demo. It combines a React/Vite interface, a FastAPI prediction API, and a TF-IDF + Logistic Regression classifier trained from the ISOT Fake News dataset.

> **Important:** this is a machine-learning estimate, not a fact-checking service. The ISOT benchmark is historical, English-language, and has source-related biases. Do not use its predictions as proof that a current article is true or false.

## Features

- React input interface for a required headline and optional full article body.
- Live service/model readiness status and clear setup/offline errors.
- Fake/Real model label, class probabilities, and model confidence score.
- Model evaluation snapshot: accuracy, fake-class precision, recall, and F1-score.
- FastAPI endpoints for health, model metrics, and predictions.
- Reproducible dataset cleaning/merging and model training scripts.
- Notebook that calls the exact same training and preprocessing code as the API.
- Automated Python tests for data cleaning, model training, and API behavior.

## Project structure

```text
fake-news-prediction-app/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI routes and request validation
│   │   ├── predictor.py         # Loads model and performs predictions
│   │   └── preprocessing.py     # Shared stemming/stop-word pipeline
│   ├── data/
│   │   ├── fake.csv.zip         # User-provided fake-news CSV archive
│   │   ├── fake_source_inspection.json # Quality checks on the supplied fake file
│   │   ├── True.csv             # Retrieved real-news CSV (not bundled)
│   │   ├── train.csv            # Combined, cleaned dataset (generated)
│   │   └── dataset_info.json    # Cleaning and label-count report (generated)
│   ├── models/
│   │   ├── fake_news_model.joblib # Trained model (generated)
│   │   └── metrics.json           # Evaluation metrics (generated)
│   ├── prepare_dataset.py      # Download, clean, label and merge data
│   ├── train_model.py          # Train/evaluate/export model
│   └── requirements.txt
├── frontend/
│   ├── src/App.jsx              # React application
│   ├── src/api.js               # FastAPI client
│   └── src/styles.css
├── notebooks/
│   └── Project_5_Fake_News_Prediction.ipynb
├── tests/                       # pytest checks
├── requirements-dev.txt
└── README.md
```

## Dataset

The supplied `backend/data/fake.csv.zip` contains 23,481 fake-news articles with columns `title`, `text`, `subject`, and `date`. The matching University of Victoria ISOT archive contains 21,417 real-news articles. The original benchmark therefore has 44,898 rows before cleaning.

The dataset builder creates the following normalized schema:

| Column | Description |
|---|---|
| `title` | News headline |
| `text` | Full article text |
| `subject` | Topic/category when available |
| `date` | Publication date when available |
| `label` | `0 = Real News`, `1 = Fake News` |

The builder removes rows with empty titles or bodies, removes copies with conflicting labels, removes repeated normalized title/body pairs, shuffles deterministically, and writes `dataset_info.json` with actual counts.

### Build the combined dataset

The matching real-news file is **not included** in this package. On a computer with internet access, run the following from the project root:

```bash
python backend/prepare_dataset.py
```

The script uses the supplied `fake.csv.zip` and downloads the official archive from the ISOT Research Lab at the University of Victoria. It extracts `True.csv`, saves it under `backend/data/`, and generates `backend/data/train.csv`.

Official dataset page: <https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/>  
Direct archive used by the script: <https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip>

If the automatic download fails, download the official archive manually and run:

```bash
python backend/prepare_dataset.py --source-zip path/to/News-_dataset.zip
```

Or place the official `True.csv` at `backend/data/True.csv` and run the default command. You can also provide explicit source paths:

```bash
python backend/prepare_dataset.py --fake path/to/fake.csv.zip --true path/to/True.csv
```

**Do not create synthetic real-news records or relabel fake examples as real.** The two classes must come from the source data.

## Setup and run

Use Python 3.10+ and Node.js 18+.

For a one-command setup on a machine with internet access, run `scripts/setup_and_train.ps1` in Windows PowerShell or `bash scripts/setup_and_train.sh` on macOS/Linux. These scripts create a Python environment, install dependencies, download/prepare the official data, train the model, and install React dependencies. After setup, launch both services with `scripts/start_project.bat` (Windows) or `bash scripts/start_project.sh` (macOS/Linux).

### 1. Create a Python environment

**Windows PowerShell:**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 2. Prepare data and train the model

```bash
python backend/prepare_dataset.py
python backend/train_model.py
```

The model is saved to `backend/models/fake_news_model.joblib`. Evaluation metrics are saved to `backend/models/metrics.json` and printed in the terminal.

Training uses:

- Features: concatenated headline and full article body (`title + text`).
- Preprocessing: lowercase, alphabetic cleanup, English stop-word removal (negations retained), and Porter stemming.
- Vectorization: TF-IDF word unigrams and bigrams, maximum 100,000 features.
- Classifier: Logistic Regression.
- Evaluation: stratified 80/20 hold-out split, accuracy, fake-class precision/recall/F1, ROC-AUC, classification report, and confusion matrix.
- Leakage control: the train/test split happens before fitting the TF-IDF vocabulary.

Optional training arguments:

```bash
python backend/train_model.py --test-size 0.2 --random-state 42 --max-features 100000
```

### 3. Start FastAPI

From the project root, with the virtual environment active:

```bash
uvicorn app.main:app --reload --app-dir backend --host 127.0.0.1 --port 8000
```

Useful endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API welcome message |
| `GET` | `/api/health` | Service and model status |
| `GET` | `/api/metrics` | Trained-model evaluation metrics |
| `POST` | `/api/predict` | Predict Fake/Real from a headline and optional body |
| `GET` | `/docs` | Interactive Swagger documentation |

Example request:

```json
{
  "title": "Example news headline",
  "text": "Optional full article body goes here."
}
```

The response includes `prediction` (`0` or `1`), `label`, `probability_real`, `probability_fake`, and `confidence`. If the model is missing, the API returns HTTP 503 with setup instructions.

### 4. Start React

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

If the API runs on a different URL, create `frontend/.env` based on `.env.example`:

```env
VITE_API_URL=http://localhost:8000
```

For a production frontend build:

```bash
cd frontend
npm run build
npm run preview
```

Set `FRONTEND_ORIGINS` on the backend to a comma-separated list of allowed browser origins when deploying the frontend and API on separate hosts.

## Run the notebook

Open `notebooks/Project_5_Fake_News_Prediction.ipynb` in Jupyter after preparing the dataset. It imports the same `train_model.py`, `app.preprocessing`, and `app.predictor` modules used by FastAPI. The notebook therefore does not implement a separate or incompatible model pipeline.

## Run tests

Install the development/test dependencies:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests use small in-memory fixtures only to validate cleaning, training, and API code paths. They are not used as training data and do not claim real-world model accuracy. The notebook requires Jupyter and Matplotlib; both are included in `requirements-dev.txt`. See `MODEL_CARD.md` for intended use, dataset risks, and limitations.

## Limitations, bias and responsible use

- The ISOT corpus is historical and primarily covers US politics/world news, mostly from 2016–2017.
- In this collection, the truthful class was obtained from Reuters while fake articles were collected from sources flagged as unreliable by the dataset authors. A model can learn publisher, dateline, or writing-style cues instead of assessing factual claims.
- A random hold-out split can overstate real-world generalization. The reported scores describe only the chosen split of this dataset; they do not establish performance on current news, unseen publishers, other languages, or different domains.
- Logistic Regression probabilities are scores, not automatically calibrated probabilities of truth.
- Do not use the prediction as the sole basis for consequential decisions. Verify claims against primary sources and independent fact-checkers.

## Troubleshooting

**`Dataset not found`** — Run `python backend/prepare_dataset.py` first and check that `backend/data/train.csv` was created.

**Dataset download fails** — Download the archive from the official source page, then pass its location using `--source-zip path/to/News-_dataset.zip`.

**API reports `model_ready: false`** — Run `python backend/train_model.py` after building the combined dataset, then restart FastAPI.

**React says API offline** — Start the FastAPI server on port 8000 and confirm `VITE_API_URL` in `frontend/.env`.

**CORS error when deployed** — Set `FRONTEND_ORIGINS` to the exact frontend origin(s), separated by commas, then restart the API.
