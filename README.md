# Bank Marketing Campaign Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-teal.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)

## 🎯 Project Overview

A machine learning solution designed to predict the success of bank marketing campaigns for term deposit subscriptions. This project implements a Random Forest Classifier achieving 89% accuracy in predicting whether a client will subscribe to a term deposit based on various demographic and behavioral features.

### 🔑 Key Features

- **Advanced Data Preprocessing**: Robust pipeline handling numerical and categorical features
- **Machine Learning Model**: Optimized Random Forest Classifier with SMOTE for handling class imbalance
- **RESTful API**: FastAPI-based endpoint for real-time predictions
- **Comprehensive Analysis**: Detailed Jupyter notebooks covering EDA, preprocessing, and modeling
- **Production-Ready**: Scalable architecture with proper error handling and logging

## 🗂️ Project Structure

```plaintext
├── data/                                   # Dataset files and processed data
│   ├── submission/real_submission.csv      # Example submission format
│   ├── test/test.csv                       # Test dataset
│   ├── row/train.csv                       # Training dataset
│   └──  processed/train_preprocessed.csv   # Preprocessed training data
├── deployment/                             # Deployment configurations
│   └── app.py                              # FastAPI application
├── models/                                 # Saved model artifacts
│   ├── feature_names.joblib                # Feature names for prediction
│   └── model_pipeline.joblib               # Trained model pipeline
├── notebooks/                              # Jupyter notebooks
│   ├── 01_EDA.ipynb                        # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb              # Data preprocessing steps
│   └── 03_Modeling.ipynb                   # Model development and evaluation
├── report/                                 # Project documentation
│    └──  Bank_Marketing_Campaign_Report.md
└── src/                                    # Source code
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
- Virtual environment (recommended)

### Installation

1. Clone the repository:

```bash
git clone {path_to_repository}
cd Bank-Marketing-Prediction
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

To train a new model:

```bash
python src/train_model.py
```

This will:
- Load and preprocess the training data
- Train the Random Forest model with optimized hyperparameters
- Save the model pipeline and feature names
- Generate performance metrics

### Making Predictions

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
        "balance": 2343,
        "housing": "yes",
        "duration": 123,
        "campaign": 3,
        "pdays": 999,
        "previous": 0,
        "contact": "cellular",
        "month": "may",
        "day": 5,
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
  - Output: Binary prediction (0/1)
  - Response time: < 100ms