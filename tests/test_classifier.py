# tests/test_classifier.py
import numpy as np
from app.classifier import classify_batch

def test_classify_batch_returns_one_per_image():
    # Construct a low-res tracking batch of two empty blank images
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    results = classify_batch(images)
    # Assert validation rule: 2 images in should return exactly 2 results out
    assert len(results) == 2

def test_classify_batch_result_shape():
    # Construct a low-res verification tracking batch container
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    result = classify_batch(images)[0]
    # Assert validation rule: Ensure the output contains expected key-value maps
    assert "prediction" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)