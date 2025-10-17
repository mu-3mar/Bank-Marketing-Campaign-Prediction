import pandas as pd
import numpy as np
import joblib
import os
from preprocess import preprocess_data

class BankMarketingPredictor:
    def __init__(self, model_path='../models/model_pipeline.joblib', feature_names_path='../models/feature_names.joblib'):
        """
        Initialize the predictor with saved model and feature names.
        
        Args:
            model_path: Path to the saved model pipeline
            feature_names_path: Path to the saved feature names
        """
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)
        
    def predict(self, data, return_proba=True):
        """
        Make predictions on input data.
        
        Args:
            data: DataFrame or dict containing input features
            return_proba: Whether to return probability scores
            
        Returns:
            DataFrame with SampleId, predicted class, and probability
        """
        # Store original SampleId if present
        sample_ids = None
        if isinstance(data, pd.DataFrame) and 'SampleId' in data.columns:
            sample_ids = data['SampleId'].copy()
        elif isinstance(data, dict) and 'SampleId' in data:
            sample_ids = pd.Series([data['SampleId']])
            
        # Preprocess the data
        is_single_sample = isinstance(data, dict)
        preprocessed_data = preprocess_data(data, is_single_sample=is_single_sample)
        
        # Ensure features are in the correct order
        preprocessed_data = preprocessed_data[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(preprocessed_data)
        probabilities = self.model.predict_proba(preprocessed_data)[:, 1] if return_proba else None
        
        # Create results DataFrame
        if sample_ids is None:
            sample_ids = pd.Series(range(len(predictions)))
            
        results = pd.DataFrame({
            'SampleId': sample_ids,
            'subscribed': predictions
        })
        
        if return_proba:
            results['probability'] = probabilities
            
        return results

def predict_file(input_path, output_path=None, model_path='../models/model_pipeline.joblib', 
                feature_names_path='../models/feature_names.joblib'):
    """
    Make predictions on a CSV file.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save predictions (optional)
        model_path: Path to saved model
        feature_names_path: Path to saved feature names
    
    Returns:
        DataFrame with predictions
    """
    # Load data
    data = pd.read_csv(input_path)
    
    # Initialize predictor
    predictor = BankMarketingPredictor(model_path, feature_names_path)
    
    # Make predictions
    predictions = predictor.predict(data)
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
    
    return predictions

def get_single_sample_input():
    """Get user input for a single sample"""
    print("\nPlease enter the following information:")
    
    sample = {}
    
    # Get SampleId
    sample['SampleId'] = int(input("Enter SampleId (integer): "))
    
    # Get age
    while True:
        try:
            age = int(input("Enter age (18-95): "))
            if 18 <= age <= 95:
                sample['age'] = age
                break
            print("Age must be between 18 and 95")
        except ValueError:
            print("Please enter a valid number")
    
    # Get categorical inputs
    sample['job'] = input("Enter job (admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown): ").lower()
    sample['marital'] = input("Enter marital status (married, single, divorced): ").lower()
    sample['education'] = input("Enter education (primary, secondary, tertiary, unknown): ").lower()
    sample['default'] = input("Enter default (yes/no): ").lower()
    sample['housing'] = input("Enter housing (yes/no): ").lower()
    sample['loan'] = input("Enter loan (yes/no): ").lower()
    sample['contact'] = input("Enter contact (telephone, cellular, unknown): ").lower()
    
    # Get numerical inputs
    sample['balance'] = float(input("Enter balance: "))
    sample['day'] = int(input("Enter day (1-31): "))
    sample['month'] = input("Enter month (jan-dec): ").lower()
    sample['duration'] = int(input("Enter duration (in seconds): "))
    sample['campaign'] = int(input("Enter campaign: "))
    sample['pdays'] = int(input("Enter pdays: "))
    sample['previous'] = int(input("Enter previous: "))
    sample['poutcome'] = input("Enter poutcome (failure, other, success, unknown): ").lower()
    
    return sample

if __name__ == "__main__":
    print("Welcome to Bank Marketing Predictor!")
    print("===================================")
    
    while True:
        print("\nChoose prediction mode:")
        print("1. Predict from CSV file")
        print("2. Predict single sample")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            # File prediction
            input_path = input("\nEnter the path to input CSV file: ")
            output_path = input("Enter the path for output CSV file (or press Enter to skip saving): ").strip()
            
            output_path = output_path if output_path else None
            
            try:
                predictions = predict_file(input_path, output_path)
                print("\nPredictions:")
                print(predictions)
                
            except FileNotFoundError:
                print(f"\nError: File '{input_path}' not found!")
            except Exception as e:
                print(f"\nError: {str(e)}")
                
        elif choice == '2':
            # Single sample prediction
            try:
                sample = get_single_sample_input()
                predictor = BankMarketingPredictor()
                prediction = predictor.predict(sample)
                
                print("\nPrediction Result:")
                print(f"SampleId: {prediction['SampleId'].iloc[0]}")
                print(f"Subscribed: {'Yes' if prediction['subscribed'].iloc[0] else 'No'}")
                print(f"Probability: {prediction['probability'].iloc[0]:.2%}")
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                
        elif choice == '3':
            print("\nThank you for using Bank Marketing Predictor!")
            break
            
        else:
            print("\nInvalid choice! Please enter 1, 2, or 3.")