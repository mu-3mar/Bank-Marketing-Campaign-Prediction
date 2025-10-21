# Bank Marketing Campaign Analysis Report

## Executive Summary

This report presents a comprehensive analysis of a bank marketing campaign prediction project. The goal was to develop a machine learning model to predict whether a client would subscribe to a term deposit based on various demographic and behavioral features. The final model achieves strong predictive performance while addressing the inherent class imbalance in the dataset.

## 1. Problem Statement

The bank aims to optimize its marketing campaigns by identifying customers most likely to subscribe to term deposits. This helps in:
- Reducing marketing costs by targeting the right customers
- Improving campaign efficiency
- Enhancing customer satisfaction through more relevant targeting

## 2. Data Analysis

### 2.1 Dataset Overview
- Total samples: ~22,000 customer records
- Features: 17 input variables
- Target: Binary (subscribe: yes/no)
- No missing values
- Significant class imbalance (11% positive class)

### 2.2 Key Features
1. **Customer Demographics**
   - Age (18-95 years)
   - Job type
   - Education level
   - Marital status

2. **Financial Indicators**
   - Account balance
   - Housing loan status
   - Previous loans

3. **Campaign Information**
   - Contact type (cellular/telephone)
   - Month of contact
   - Call duration
   - Number of contacts
   - Previous campaign outcome

### 2.3 Feature Engineering
Several feature transformations were implemented:
- Age binning into meaningful categories
- Logarithmic transformation of numeric features (balance, duration, campaign)
- Cyclical encoding of month feature
- One-hot encoding of categorical variables
- Ordinal encoding of education levels

## 3. Methodology

### 3.1 Data Preprocessing Pipeline
1. Feature selection and engineering
2. Handling class imbalance using SMOTE
3. Feature scaling using RobustScaler
4. Train-test split (90-10)

### 3.2 Model Development
The final model uses a Random Forest Classifier with the following optimized parameters:
- n_estimators: 168
- max_depth: 14
- min_samples_split: 9
- min_samples_leaf: 4
- max_features: 'sqrt'
- bootstrap: False

## 4. Results and Performance

### 4.1 Model Performance Metrics
- Accuracy: 89%
- Precision: 57%
- Recall: 60%
- F1-Score: 58%
- ROC-AUC: 0.77

### 4.2 Feature Importance
Top influential features:
1. Call duration
2. Month of contact
3. Age
4. Balance
5. Previous campaign outcome

## 5. Deployment

The model has been deployed as a REST API using FastAPI, enabling:
- Real-time predictions
- Easy integration with existing systems
- Scalable architecture
- Automated preprocessing pipeline

### 5.1 API Endpoints
- POST `/predict`: Returns prediction for a single customer
- Response time: < 100ms

## 6. Limitations and Future Improvements

### 6.1 Current Limitations
- Model requires call duration (only available after contact)
- Limited to features available at time of collection
- Seasonal patterns may change over time

### 6.2 Recommended Improvements
1. Develop a pre-contact scoring model
2. Implement model monitoring and retraining pipeline
3. Collect additional relevant features

## 7. Conclusions

The developed solution provides a robust framework for predicting term deposit subscriptions. Key achievements:
- Successful handling of class imbalance
- Strong predictive performance
- Production-ready API implementation
- Comprehensive preprocessing pipeline

The model can significantly improve campaign efficiency by enabling targeted marketing approaches and reducing unnecessary customer contacts.

## 8. Technical Stack

- Python 3.8+
- Key Libraries:
  - scikit-learn
  - pandas
  - numpy
  - FastAPI
  - imbalanced-learn
  - joblib

## Appendix: Implementation Details

### A1. Model Pipeline Components
```python
ImbPipeline([
    ('smote', SMOTE(random_state=74, k_neighbors=5)),
    ('scaler', RobustScaler()),
    ('model', RandomForestClassifier(
        n_estimators=168,
        max_depth=14,
        max_features='sqrt',
        min_samples_split=9,
        min_samples_leaf=4,
        bootstrap=False,
        random_state=74
    ))
])
```

### A2. Feature Engineering Details
```python
# Age binning
age_bins = [17, 29, 45, 60, 95]
age_labels = ['young', 'adult', 'middle_aged', 'senior']

# Numeric transformations
balance = np.log1p(balance - min_balance)
duration = np.log1p(duration)
campaign = np.log1p(campaign)

# Cyclical month encoding
month_sin = np.sin(2 * np.pi * month_num / 12)
month_cos = np.cos(2 * np.pi * month_num / 12)
```