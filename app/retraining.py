import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from models import SessionLocal, Prediction

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

"""
Theoretical skeleton of a potential retraining pipeline.
Not inplemented in the main application because out of scope.
"""

def fetch_original_training_data():
    RANDOM_STATE = 42
    X, y = fetch_openml("mnist_784", version=1,return_X_y=True, as_frame=False)

    # Remove the zeros:
    mask = y != "0"
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)
    
    return X_train, X_test, y_train, y_test

def fetch_reviewed_hitl_data():
    db = SessionLocal()
    
    try:
        reviewed_records = db.query(Prediction).filter(Prediction.is_reviewed == True).all()
        
        X_list = []
        y_list = []
        
        for record in reviewed_records:
            pixel_array = np.array(record.pixels, dtype=np.uint8)
            X_list.append(pixel_array)
            y_list.append(record.human_label)
            
    finally:
        db.close()

    # Safety check: Prevent numpy stacking errors if the DB is empty
    if not X_list:
        print("Warning: No reviewed HITL data found in the database.")
        return np.empty((0, 28, 28), dtype=np.uint8), np.array([])

    # Stack the list of 2D arrays into a single 3D array of shape (N, 28, 28)
    X_new = np.stack(X_list)
    y_new = np.array(y_list)

    print(f"Successfully loaded {len(X_new)} human-reviewed samples from the database.")
    
    return X_new, y_new

def execute_concept_expansion(version: str):
    print("Initiating V2 Retraining Pipeline: Concept Expansion (9 -> 10 Classes)")
    # 1. Gather all data
    X_base, y_base = fetch_original_training_data()
    X_hitl, y_hitl = fetch_reviewed_hitl_data()

    # Combine datasets
    X_combined = np.concatenate([X_base, X_hitl])
    y_combined = np.concatenate([y_base, y_hitl])

    # 2. Preprocess the Data (Matching your app/classifier.py pipeline)
    # Flatten to 784 dimensions
    X_flat = X_combined.reshape(len(X_combined), -1)
    
    # Binarize for the Logistic Regression (as defined in your pipeline)
    X_binarized = (X_flat > 128).astype(float)

    print(f"Total training samples: {len(X_combined)}")
    print(f"Unique classes identified: {np.unique(y_combined)}")

    # Retrain the Logistic Regression (The Classifier)
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced") # balanced due to uneven class distributions
    clf.fit(X_binarized, y_combined)

    # 4. Retrain the Isolation Forest (The OOD Safeguard)
    print("Training Isolation Forest...")
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso.fit(X_flat)

    # 5. Export Artifacts
    print("Exporting artifacts...")
    joblib.dump(clf, f"models/pixelwise_classifier_{version}.pkl")
    joblib.dump(iso, f"models/pixelwise_iso_forest_{version}.pkl")
    
    # Safe classes to document potential model expansion
    joblib.dump(clf.classes_, f"models/classes_{version}.pkl")

if __name__ == "__main__":
    execute_concept_expansion("v2")