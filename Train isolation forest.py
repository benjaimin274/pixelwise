import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    RANDOM_STATE = 42
    CLASSES = ["1","2","3","4","5","6","7","8","9"]
    X, y = fetch_openml("mnist_784", version=1,return_X_y=True, as_frame=False)

    # Remove the zeros:
    mask = y != "0"
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)

    # Reshape into the correct format
    # ! Input shape is really important since it is dependent on the inputs in the pipeline...
    X_train_flat = X_train.reshape(len(X_train), -1)

    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination='auto', 
        random_state=42,
    )

    print("Training Isolation Forest (this may take a minute)...")
    iso_forest.fit(X_train_flat)

    # Save the model
    model_path = "models/isolation_forest_v1.pkl"
    joblib.dump(iso_forest, model_path)
    print(f"Model successfully saved to {model_path}")