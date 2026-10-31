import pandas as pd
import joblib
from pathlib import Path

from src.preprocess import preprocess_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class BankMarketingPredictor:
    def __init__(self, model_path=None, feature_names_path=None):
        """
        Initialize the predictor with saved model and feature names.
        
        Args:
            model_path: Path to the saved model pipeline
            feature_names_path: Path to the saved feature names
        """
        if model_path is None or feature_names_path is None:
            # Get the absolute path of the project root directory
            model_path = PROJECT_ROOT / "models" / "model_pipeline.joblib"
            feature_names_path = PROJECT_ROOT / "models" / "feature_names.joblib"
        
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
        
        # Create results DataFrame
        if sample_ids is None:
            sample_ids = pd.Series(range(len(predictions)))
            
        results = pd.DataFrame({
            'SampleId': sample_ids,
            'y': predictions.astype(int)  # Ensure predictions are integers
        })
            
        return results

def predict_file(input_path, output_path=None, model_path=None,
            feature_names_path=None):
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
    
    model_path = model_path or PROJECT_ROOT / "models" / "model_pipeline.joblib"
    feature_names_path = feature_names_path or PROJECT_ROOT / "models" / "feature_names.joblib"

    predictor = BankMarketingPredictor(model_path, feature_names_path)
    
    # Make predictions
    predictions = predictor.predict(data)
    
    # Save if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
    
    return predictions

if __name__ == "__main__":
    print("Bank Marketing Predictor")
    print("=======================")
    
    # Get the absolute path of the project root directory
    default_input = PROJECT_ROOT / "data" / "raw" / "test.csv"
    default_output = PROJECT_ROOT / "data" / "submission" / "real_submission.csv"
    
    # Get input path from user
    print("\nInput File:")
    print(f"Default path: {default_input}")
    user_input = input("Press Enter to use default or enter new path: ").strip()
    input_path = user_input if user_input else default_input
    
    # Get output path from user
    print("\nOutput File:")
    print(f"Default path: {default_output}")
    user_output = input("Press Enter to use default or enter new path: ").strip()
    output_path = user_output if user_output else default_output
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("\nUsing paths:")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("\nProcessing...")
    
    try:
        # Make predictions
        predictions = predict_file(input_path, output_path)
        
        # Print summary
        positives = int(predictions['y'].sum())
        negatives = int((predictions['y'] == 0).sum())
        total = len(predictions)
        
        print("\nPrediction completed successfully!")
        print(f"Total predictions: {total}")
        print(f"Positive predictions (y=1): {positives:,d} ({(positives/total)*100:.1f}%)")
        print(f"Negative predictions (y=0): {negatives:,d} ({(negatives/total)*100:.1f}%)")
        print(f"\nResults saved to: {output_path}")
        print("Format: SampleId,y")
        
    except FileNotFoundError:
        print(f"\nError: Input file '{input_path}' not found!")
    except Exception as e:
        print(f"\nError: {str(e)}")