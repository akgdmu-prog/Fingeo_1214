import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_and_prepare_data(matrix_path):
    """
    Loads the synthetic dataset matrix, extracts the engineering features
    including quality control indicators, and prepares matrices for XGBoost.
    """
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Missing training matrix at: {matrix_path}. Please create the file first.")
        
    with open(matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Convert JSON array of objects directly into a Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Define your Feature Columns (X) explicitly matching your pipeline output
    feature_cols = [
        "vegetation_index",
        "flood_proximity_score",
        "extreme_wind_count",
        "infrastructure_hazard",
        "urbanization_index",
        "slope_gradient",
        "snowfall_risk",
        "soil_drainage_ratio",
        "crime_risk_index",
        "is_soil_fallback",    # Native model flag for fallback handling
        "is_crime_fallback"    # Native model flag for fallback handling
    ]
    
    # Define your Target Column (y)
    target_col = "target_risk_score"
    
    X = df[feature_cols]
    y = df[target_col]
    location_ids = df["location_id"]
    
    return X, y, location_ids

def train_risk_model():
    # Dynamic path configuration pointing to your dataset folder named 'dataset'
    matrix_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "training_matrix.json")
    
    print("[XGBOOST] Loading training matrix data...")
    X, y, locations = load_and_prepare_data(matrix_file)
    
    # Split dataset into 80% training and 20% validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"[XGBOOST] Training set size: {X_train.shape[0]} locations")
    print(f"[XGBOOST] Validation set size: {X_val.shape[0]} locations")
    
    # Initialize XGBoost Regressor optimized for small, multi-hazard tabular spaces
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
    
    print("[XGBOOST] Fitting decision tree matrix structures...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate performance
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    
    print("\n" + "="*40)
    print("      MODEL TRAINING PERFORMANCE")
    print("="*40)
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"R-squared Score (R²):      {r2:.4f}")
    print("="*40)
    
    # Display feature importance vectors
    print("\n[XGBOOST] Relative Feature Weights Calculated by Trees:")
    importances = model.feature_importances_
    for col, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
        print(f" - {col:<25}: {imp:.4f}")
        
    # Save the serialization model weight file safely
    model_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_xgboost_model.json")
    model.save_model(model_output_path)
    print(f"\n[SUCCESS] Model serialized and saved to: {model_output_path}")

if __name__ == "__main__":
    train_risk_model()