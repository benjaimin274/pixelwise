import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.classifier import classify_batch
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

"""
Local experiment to calculate the FPR95 threshold per class.
Collects confidence scores for each class from correctly classified predictions
and returns the lower 5% confidence score for each class.
"""

if __name__ == "__main__":
    RANDOM_STATE = 42
    CLASSES = ["1","2","3","4","5","6","7","8","9"]
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    # Remove the zeros:
    mask = y != "0"
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)

    # Reshape into the correct format
    images_test = X_test.reshape(-1, 28, 28).astype(np.uint8)

    # Perform the classification on the entire batch
    results = classify_batch(images_test)

    # Collect confidence scores per class for correctly classified predictions
    class_confidences = {cls: [] for cls in CLASSES}

    for pred, true_label in zip(results, y_test):
        if str(pred["prediction"]) == str(true_label):
            class_confidences[str(true_label)].append(pred["confidence"])

    # Calculate FPR95 threshold (5th percentile) for each class
    fpr95_thresholds = {}
    for cls in CLASSES:
        if len(class_confidences[cls]) > 0:
            fpr95_thresholds[cls] = np.percentile(class_confidences[cls], 5)
        else:
            fpr95_thresholds[cls] = None

    print("FPR95 Thresholds per class:")
    for cls in CLASSES:
        if fpr95_thresholds[cls] is not None:
            print(f"Class {cls}: {fpr95_thresholds[cls]:.4f}")
        else:
            print(f"Class {cls}: No correct predictions")

    print(f"\nThreshold dictionary:\n{fpr95_thresholds}")
