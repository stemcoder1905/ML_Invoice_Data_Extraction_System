"""
Automated Unit Tests for Image Preprocessing Pipeline.
"""

import numpy as np
import cv2

from src.image_preprocessing import ImagePreprocessor


def test_grayscale_conversion():
    preprocessor = ImagePreprocessor()
    # 3-channel BGR
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = preprocessor.to_grayscale(bgr)
    assert len(gray.shape) == 2

    # Already grayscale
    gray_again = preprocessor.to_grayscale(gray)
    assert len(gray_again.shape) == 2


def test_resize_aspect_ratio():
    preprocessor = ImagePreprocessor()
    # Image larger than max dimension
    large_img = np.ones((4000, 2000, 3), dtype=np.uint8) * 255
    resized, scale = preprocessor.resize_image(large_img, max_dimension=2000)
    assert scale < 1.0
    assert resized.shape[0] == 2000
    assert resized.shape[1] == 1000


def test_deskew_and_binarization():
    preprocessor = ImagePreprocessor()
    # Create white canvas with black text line
    img = np.ones((300, 500), dtype=np.uint8) * 255
    cv2.putText(img, "INVOICE NUMBER: 12345", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    binary = preprocessor.binarize(img, method="otsu")
    assert binary.shape == img.shape
    # Check that text region is black/white
    assert np.all(np.isin(binary, [0, 255]))

    # Test complete preprocess pipeline
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    processed, meta = preprocessor.preprocess(bgr_img)
    assert processed is not None
    assert "original_size" in meta
    assert "skew_angle_deg" in meta
