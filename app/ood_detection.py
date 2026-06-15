import os
import joblib
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Load the isolation forest model
iso_forest = joblib.load(os.getenv("MODEL_PATH_ISO_FOREST"))

def detect_anomalies_batch(images: np.ndarray) -> list[float]:
    """
    Calculates the Isolation Forest anomaly scores for a batch of images.
    """
    # Data Structure Check
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(
            f"Invalid shape for Isolation Forest batch input. "
            f"Expected (N, 28, 28), but got {images.shape}."
        )

    # Flatten the batch for the Isolation Forest (N, 784)
    arr = images.reshape(len(images), -1)
    # 3. Calculate the structural anomaly scores
    scores = iso_forest.decision_function(arr)
    return [float(score) for score in scores]

def detect_anomaly(image: np.ndarray) -> float:
    """
    Helper function to calculate the anomaly score for a single image.
    """
    return detect_anomalies_batch(image[np.newaxis])[0]