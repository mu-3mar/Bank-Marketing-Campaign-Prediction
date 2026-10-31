# Bank Marketing Campaign Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-teal.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)
[![Docker Image](https://img.shields.io/docker/v/mu3mar/bank-marketing-api?label=Docker%20Image)](https://hub.docker.com/r/mu3mar/bank-marketing-api)

## 🎯 Project Overview

A machine learning solution designed to predict the success of bank marketing campaigns for term deposit subscriptions. This project implements a Random Forest Classifier achieving 89% accuracy in predicting whether a client will subscribe to a term deposit based on various demographic and behavioral features.

### 🔑 Key Features

- **Advanced Data Preprocessing**: Robust pipeline handling numerical and categorical features
- **Machine Learning Model**: Optimized Random Forest Classifier with SMOTE for handling class imbalance
- **RESTful API**: FastAPI-based endpoint for real-time predictions
- **Interactive Web GUI**: Static HTML/CSS/JavaScript frontend with random inputs, probability meter, and prediction result
- **Comprehensive Analysis**: Detailed Jupyter notebooks covering EDA, preprocessing, and modeling
- **Dockerized Deployment**: Ready-to-run Docker image published on [Docker Hub](https://hub.docker.com/r/mu3mar/bank-marketing-api)
- **Production-Ready**: Scalable architecture with proper error handling and logging

## 🗂️ Project Structure

```plaintext
├── data/                                   # Dataset inputs, outputs, and documentation
│   ├── raw/                                # Downloaded train/test CSV files
│   ├── processed/                          # Notebook-generated datasets
│   ├── submission/real_submission.csv      # Example submission
│   └── README.md                           # Dataset source and download steps
├── frontend/                               # User-facing web interface
│   ├── index.html                           # Application page
│   ├── app.js                               # Form and API integration
│   └── styles.css                           # Responsive styling
├── models/                                 # Saved model artifacts
│   ├── feature_names.joblib                # Feature names for prediction
│   └── model_pipeline.joblib               # Trained model pipeline
├── notebooks/                              # Jupyter notebooks
│   ├── 01_EDA.ipynb                        # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb              # Data preprocessing steps
│   └── 03_Modeling.ipynb                   # Model development and evaluation
├── reports/                                # Project documentation
│   └── Bank_Marketing_Report.md
└── src/                                    # Source code
    ├── __init__.py                          # Source package marker
    ├── api.py                              # FastAPI endpoints
    ├── predict.py                          # Prediction utilities
    ├── preprocess.py                       # Data preprocessing functions
    └── train_model.py                      # Model training script
```

## 📊 Model Performance

| Metric    | Score |
|-----------|-------|
| Accuracy  | 89%   |
| Precision | 57%   |
| Recall    | 60%   |
| F1-Score  | 58%   |
| ROC-AUC   | 0.77  |

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone {path_to_repository}
cd Bank-Marketing-Prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The full `requirements.txt` includes the dependencies used for training,
notebooks, analysis, and the API. Docker installs only the API runtime
dependencies from `requirements-docker.txt` to keep the image smaller.

### Docker

The API is Dockerized and the production image is available on
[Docker Hub](https://hub.docker.com/r/mu3mar/bank-marketing-api).

Pull and run the published image:

```bash
docker pull mu3mar/bank-marketing-api:latest
docker run --rm -p 8000:8000 mu3mar/bank-marketing-api:latest
```

The API will be available at `http://localhost:8000`. The interactive frontend
can then be served separately from the project root:

```bash
python3 -m http.server 5500 --directory frontend
```

To build the image locally instead:

```bash
docker build -t bank-marketing-api .
```

## 💻 Usage

### Training the Model

To train a new model:

```bash
python -m src.train_model
```

This will:
- Load and preprocess the training data
- Train the Random Forest model with optimized hyperparameters
- Save the model pipeline and feature names
- Generate performance metrics

### Making Predictions

#### Using the Web GUI

Start the API in one terminal:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Then serve the static frontend in another terminal from the project root:

```bash
python3 -m http.server 5500 --directory frontend
```

Open `http://localhost:5500`, fill in the client information, and click
**Predict**. The **Randomise inputs** button creates a sample for a quick demo.

#### Using the API

1. Start the FastAPI server:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

2. Send predictions requests:

```python
import requests

data = {
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

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## 📋 Dataset Description

The dataset contains client information and campaign results from a Portuguese banking institution. Key features include:

- **Demographics**: age, job, marital status, education
- **Financial**: account balance, existing loans
- **Campaign**: contact type, duration, number of contacts
- **Context**: economic indicators, previous campaign results

## 🛠️ Technical Details

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Preprocessing**: 
  - Feature scaling using RobustScaler
  - SMOTE for handling class imbalance
  - Custom feature engineering pipeline
- **Hyperparameters**:
  - n_estimators: 168
  - max_depth: 14
  - min_samples_split: 9
  - bootstrap: False

### API Endpoints

- `POST /predict`
  - Input: JSON with client features
  - Output: Binary prediction (0/1) and subscription probability
  - Response time: < 100ms