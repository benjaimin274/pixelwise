import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

def fetch_training_and_test_data():
    RANDOM_STATE = 42 # recreate the same splits
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    # Remove the zeros:
    mask = y != "0"
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)
    
    # Flatten the images for input compatability:
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    return X_train_flat, X_test_flat, y_train, y_test
    
def train_forest(X_train_flat: np.ndarray) -> None:
    # Ensure that the training input for the isolation forest has the right shape:
    if (len(X_train_flat.shape) != 2) or (X_train_flat.shape[-1] != 784):
        raise ValueError(
            f"Invalid shape for Isolation Forest training input. "
            f"Expected a 2D array of shape (N, 784), but got {X_train_flat.shape}."
        )

    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination='auto', 
        random_state=42,
    )
    iso_forest.fit(X_train_flat)
    print("Traind the isolation forest successfully!")

    model_path = "models/isolation_forest_v1.pkl"
    os.makedirs("models", exist_ok= True) # Create the folder if it doesn't already exist.
    joblib.dump(iso_forest, model_path)
    print(f"model saved to {model_path}")

def fpr95_score(model_path: str, X_test_flat: np.ndarray) -> None:
    iso_forest = joblib.load(model_path)

    if (len(X_test_flat.shape) != 2) or (X_test_flat.shape[-1] != 784):
        raise ValueError(
            f"Invalid shape for Isolation Forest training input. "
            f"Expected a 2D array of shape (N, 784), but got {X_test_flat.shape}."
        )
    
    iso_scores = iso_forest.decision_function(X_test_flat)

    # Calc the FPR95 (like with the other threshhold)
    iso_threshold = np.percentile(iso_scores, 5)
    print(f"The mathematically true Isolation Forest threshold is: {iso_threshold:.4f}")

if __name__ == "__main__":
    X_train_flat, X_test_flat, y_train, y_test = fetch_training_and_test_data()
    train_forest(X_train_flat)
