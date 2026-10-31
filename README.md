# Bank Marketing Campaign Prediction

An end-to-end machine learning project for estimating whether a bank customer
will subscribe to a term deposit after a marketing contact. The project
covers exploratory analysis, preprocessing, model comparison, training,
inference, and deployment through a FastAPI service and an interactive web
interface.

> **Live demo:** [bank-marketing-campaign-prediction-self.vercel.app](https://bank-marketing-campaign-prediction-self.vercel.app/)

> **Important:** The prediction is a statistical estimate for decision
> support. It is not a guarantee of customer behaviour and should not replace
> professional judgement.

## Project Highlights

- Exploratory Data Analysis (EDA) of a Portuguese bank marketing dataset.
- Reproducible preprocessing and feature engineering pipeline.
- Comparison of eight classification algorithms.
- Class-imbalance handling with SMOTE.
- Final Random Forest model saved as a reusable scikit-learn/imbalanced-learn
  pipeline.
- FastAPI endpoint for real-time predictions and probability scores.
- Responsive frontend with random sample generation, validation, and a
  probability meter.
- Docker and Vercel deployment support.

## Results

### Model comparison

The modeling notebook evaluated the following models on the prepared
train/test split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8377 | 0.3881 | 0.6320 | 0.4809 | 0.7487 |
| Random Forest | 0.8885 | 0.5282 | 0.5911 | 0.5579 | 0.7599 |
| XGBoost | 0.8872 | 0.5233 | 0.5836 | 0.5518 | 0.7559 |
| Gradient Boosting | 0.8704 | 0.4722 | 0.7584 | 0.5820 | 0.8220 |
| AdaBoost | 0.8669 | 0.4590 | 0.6654 | 0.5432 | 0.7798 |
| Decision Tree | 0.8585 | 0.4282 | 0.5651 | 0.4872 | 0.7316 |
| KNN | 0.8138 | 0.3480 | 0.6468 | 0.4525 | 0.7416 |
| Naive Bayes | 0.6373 | 0.1949 | 0.6543 | 0.3003 | 0.6447 |

The deployed artifact is the tuned Random Forest pipeline. Its recorded
evaluation summary is:

| Metric | Score |
|---|---:|
| Accuracy | 89% |
| Precision | 57% |
| Recall | 60% |
| F1-score | 58% |
| ROC-AUC | 0.77 |

The two tables reflect two stages of the work: the first is the notebook's
baseline model comparison, while the second is the final tuned model summary
used in the project report and deployment.

## Dataset and Problem

The dataset comes from the
[Bank Marketing Prediction Kaggle competition](https://www.kaggle.com/competitions/bank-marketing-prediction).
It contains customer, financial, and campaign information collected by a
Portuguese banking institution.

- **Training shape:** 22,605 rows × 18 columns.
- **Raw inputs:** 17 customer/campaign fields plus the target.
- **Target:** `y` — whether the customer subscribed to a term deposit.
- **Target type:** binary classification.
- **Class distribution:** imbalanced, with the positive class representing
  approximately 11% of the records.
- **Missing values:** none were found in the initial dataset inspection.

The raw dataset is not committed to this repository. Download `train.csv` and
`test.csv` from Kaggle and place them in `data/raw/`.

## Data Preparation

The preprocessing workflow is documented in
[`02_Preprocessing.ipynb`](./notebooks/02_Preprocessing.ipynb) and implemented
in [`src/preprocess.py`](./src/preprocess.py). It includes:

1. Removing identifier and excluded low-value fields (`SampleId`, `previous`,
   `default`, and `loan`).
2. Binning age into `young`, `adult`, `middle_aged`, and `senior`.
3. Applying logarithmic transformations to `balance`, `duration`, and
   `campaign`.
4. Encoding the contact month with sine/cosine cyclical features.
5. One-hot encoding job, marital status, contact type, and previous outcome.
6. Ordinal encoding education.
7. Binary encoding of housing-loan status.
8. Creating age indicator variables and enforcing a stable feature order for
   inference.
9. Applying SMOTE to the training data to address class imbalance.
10. Scaling numeric values with `RobustScaler`.

## Final Model

The saved model is a `RandomForestClassifier` inside an imbalanced-learn
pipeline:

```text
SMOTE -> RobustScaler -> RandomForestClassifier
```

Final Random Forest configuration:

```text
n_estimators     = 168
max_depth        = 14
min_samples_split = 9
min_samples_leaf = 4
max_features     = "sqrt"
bootstrap        = False
random_state     = 74
```

The trained artifacts are:

- `models/model_pipeline.joblib` — preprocessing/model pipeline.
- `models/feature_names.joblib` — feature order required at inference time.

## Repository Structure

```text
.
├── app.py                         # Vercel Python entry point
├── Dockerfile                     # Container image definition
├── requirements.txt               # Local/API dependencies
├── requirements-docker.txt        # Docker runtime dependencies
├── data/
│   ├── raw/                       # Kaggle files (not committed)
│   ├── processed/                 # Generated preprocessed datasets
│   ├── submission/                # Generated prediction CSV files
│   └── README.md                  # Dataset download instructions
├── frontend/
│   ├── index.html                 # Prediction form and result panel
│   ├── app.js                     # Form/API integration
│   ├── styles.css                 # Responsive UI styles
│   └── logo-animated.html         # Animated brand mark
├── models/
│   ├── feature_names.joblib
│   └── model_pipeline.joblib
├── notebooks/
│   ├── 01_EDA.ipynb               # Data exploration
│   ├── 02_Preprocessing.ipynb     # Feature engineering
│   └── 03_Modeling.ipynb          # Comparison and evaluation
├── reports/
│   └── Bank_Marketing_Report.md   # Detailed project report
└── src/
    ├── api.py                     # FastAPI application
    ├── predict.py                 # Batch/single prediction utilities
    ├── preprocess.py              # Inference preprocessing
    └── train_model.py             # Training and artifact export
```

## Local Setup

### Requirements

- Python 3.9 or newer.
- `pip`.
- Kaggle dataset files if you want to rerun preprocessing/training.

### Installation

```bash
git clone https://github.com/mu-3mar/Bank-marketing-prediction.git
cd Bank-marketing-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

### Run the API and frontend

The FastAPI application serves both the prediction endpoint and the static
frontend:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in a browser.

If you want to serve the frontend separately during development:

```bash
python3 -m http.server 5500 --directory frontend
```

### Train the model

After placing the raw data in `data/raw/` and creating the processed training
file through the preprocessing notebook/script:

```bash
python3 -m src.train_model
```

The command writes the trained pipeline and feature names to `models/`.

## API Usage

### `POST /predict`

Request body:

```json
{
  "data": {
    "age": 41,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 2343,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 5,
    "month": "may",
    "duration": 123,
    "campaign": 3,
    "pdays": 999,
    "previous": 0,
    "poutcome": "unknown"
  }
}
```

Example with `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":{"age":41,"job":"management","marital":"married","education":"tertiary","default":"no","balance":2343,"housing":"yes","loan":"no","contact":"cellular","day":5,"month":"may","duration":123,"campaign":3,"pdays":999,"previous":0,"poutcome":"unknown"}}'
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.1234
}
```

- `prediction`: `1` indicates a predicted subscription; `0` indicates no
  predicted subscription.
- `probability`: estimated probability of subscription, returned as a value
  between `0` and `1`.

## Docker

Run the published image:

```bash
docker pull mu3mar/bank-marketing-api:latest
docker run --rm -p 8000:8000 mu3mar/bank-marketing-api:latest
```

Or build locally:

```bash
docker build -t bank-marketing-api .
docker run --rm -p 8000:8000 bank-marketing-api
```

The API is then available at [http://localhost:8000](http://localhost:8000).
The image is published at
[Docker Hub](https://hub.docker.com/r/mu3mar/bank-marketing-api).

## Deployment

The project is configured for Vercel's native Python runtime:

- [`app.py`](./app.py) exposes the FastAPI application.
- The API and frontend are served from the same origin.
- No separate frontend deployment or rewrite configuration is required.
- Use the Python framework preset and leave the build command empty.

The current deployment is available at the
[live demo](https://bank-marketing-campaign-prediction-self.vercel.app/).

## Limitations and Responsible Use

- `duration` is only known after a contact has taken place, so this is not a
  pure pre-contact targeting model.
- Historical campaign patterns may not represent future customer behaviour.
- Class imbalance means accuracy alone is not sufficient for judging model
  quality; precision, recall, F1-score, and ROC-AUC should also be considered.
- The model is trained on one historical dataset and requires monitoring and
  retraining if the customer population or campaign strategy changes.
- Predictions should support, not automate, decisions about customer contact.

## Further Documentation

- [Detailed project report](./reports/Bank_Marketing_Report.md)
- [EDA notebook](./notebooks/01_EDA.ipynb)
- [Preprocessing notebook](./notebooks/02_Preprocessing.ipynb)
- [Modeling notebook](./notebooks/03_Modeling.ipynb)
- [Dataset instructions](./data/README.md)

## License

This repository does not currently declare a software license. Check with the
repository owner before redistributing or using it commercially.
