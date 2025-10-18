import pandas as pd
import numpy as np
import os

def preprocess_data(data, is_single_sample=False):
    """
    Preprocess the input data according to the preprocessing pipeline.
    
    Args:
        data: Can be either a pandas DataFrame or a dictionary for single sample
        is_single_sample: Boolean indicating if the input is a single sample
        
    Returns:
        Preprocessed data as pandas DataFrame
    """
    # Convert single sample dictionary to DataFrame if needed
    if is_single_sample:
        data = pd.DataFrame([data])
    else:
        data = data.copy()
        
    # Feature Selection - Drop unnecessary columns
    columns_to_drop = ['SampleId', 'previous', 'default', 'loan']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # Age binning
    data['age'] = pd.cut(data['age'], 
                        bins=[17, 29, 45, 60, 95], 
                        labels=['young', 'adult', 'middle_aged', 'senior'])
    
    # Balance transformation
    balance_min = data['balance'].min()
    data['balance'] = np.log1p(data['balance'] - balance_min)
    
    # Duration transformation
    data['duration'] = np.log1p(data['duration'])
    
    # Campaign transformation
    data['campaign'] = np.log1p(data['campaign'])
    
    # Cyclical month encoding
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    data['month_num'] = data['month'].map(month_mapping)
    data['month_sin'] = np.sin(2 * np.pi * data['month_num'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month_num'] / 12)
    data.drop(columns=['month', 'month_num'], inplace=True)
    
    # One-hot encoding
    data = pd.get_dummies(data, columns=['job', 'marital', 'contact', 'poutcome'], drop_first=True, dtype=int)
    
    # Ordinal encode education
    education_mapping = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
    data['education'] = data['education'].map(education_mapping)
    
    # Binary encode housing
    data['housing'] = data['housing'].replace({'yes': 1, 'no': 0})
    
    # Create age dummy variables
    age_dummies = pd.get_dummies(data['age'], prefix='age', drop_first=True, dtype=int)
    data = pd.concat([data, age_dummies], axis=1)
    data.drop(columns=['age'], inplace=True)

    # Select and order the required features
    required_features = [
        'education', 'balance', 'housing', 'day', 'duration', 'campaign', 'pdays',
        'month_sin', 'month_cos', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
        'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student',
        'job_technician', 'job_unemployed', 'job_unknown', 'marital_married', 'marital_single',
        'contact_telephone', 'contact_unknown', 'poutcome_other', 'poutcome_success',
        'poutcome_unknown', 'age_adult', 'age_middle_aged', 'age_senior'
    ]
    
    # Ensure all required columns exist, fill with 0 if missing
    for col in required_features:
        if col not in data.columns:
            data[col] = 0
            
    # Return only the required features in the correct order
    return data[required_features]

def preprocess_file(input_path, output_path=None):
    """
    Preprocess a CSV file containing bank marketing data.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Optional path to save the preprocessed data
        
    Returns:
        Preprocessed DataFrame
    """
    # Read the data
    data = pd.read_csv(input_path)
    
    # Preprocess the data
    preprocessed_data = preprocess_data(data, is_single_sample=False)
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        preprocessed_data.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to: {output_path}")
    
    return preprocessed_data

# Module exports
__all__ = ['preprocess_data', 'preprocess_file']
