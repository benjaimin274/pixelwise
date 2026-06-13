from app.classifier import classify_batch
import numpy as np
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
    images_test = X_test.reshape(-1, 28, 28).astype(np.uint8)

    # Perform the classification on the entire batch
    results = classify_batch(images_test)

    # Filter out all the images that have been correctly classified:
    true_predictions_msp = []

    for pred, true_label in zip(results, y_test):
            if str(pred["prediction"]) == str(true_label):
                    true_predictions_msp.append(pred["confidence"])

    thresh = np.percentile(true_predictions_msp, 5)
    print(f"The mathematically true threshold using production logic is: {thresh:.4f}")