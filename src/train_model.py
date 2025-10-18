import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier


def split_features_target(df):
    """Split DataFrame into features and target"""
    X = df.drop(columns=['subscribed'])
    y = df['subscribed']
    return X, y

def create_pipeline(random_state=74):
    """
    Create a pipeline with preprocessing, SMOTE, scaling, and Random Forest
    """
    return ImbPipeline([
        ('smote', SMOTE(random_state=random_state,k_neighbors=5)),
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

def train(X, y, random_state=74):
    """
    Train the model and evaluate its performance
    """
    
    # Create and train pipeline
    pipeline = create_pipeline(random_state=random_state)
    print("\nTraining model...")
    pipeline.fit(X, y)
    
    return pipeline

def save_model_artifacts(pipeline, feature_names, output_dir='models'):
    """Save model pipeline and feature names"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pipeline
    model_path = os.path.join(output_dir, 'model_pipeline.joblib')
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, 'feature_names.joblib')
    joblib.dump(feature_names, features_path)
    print(f"Feature names saved to: {features_path}")

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    processed_data = pd.read_csv('data/processed/train_preprocessed.csv')
    
    # Split features and target
    X, y = split_features_target(processed_data)
    print(f"Data shape after preprocessing: {X.shape}")
    
    # Train model and evaluate
    pipeline = train(X, y)
    
    # Save model artifacts
    save_model_artifacts(pipeline, X.columns)
    
    print("\nTraining pipeline completed successfully")

if __name__ == "__main__":
    main()