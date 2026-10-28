"""
Image Preprocessing Module for Invoice Documents.
Implements OpenCV-based deskewing, noise reduction, grayscale conversion,
adaptive thresholding, and resolution normalization.
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2


class ImagePreprocessor:
    """Preprocesses raw invoice images or rendered PDF pages for OCR extraction."""

    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Converts an image to single-channel grayscale if needed."""
        if len(image.shape) == 2:
            return image
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")

    def resize_image(self, image: np.ndarray, max_dimension: int = 3000, min_dimension: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Resizes image to an optimal resolution range for OCR while preserving aspect ratio.
        Returns resized image and scale factor.
        """
        h, w = image.shape[:2]
        max_side = max(h, w)
        min_side = min(h, w)

        scale = 1.0
        if max_side > max_dimension:
            scale = max_dimension / float(max_side)
        elif min_side < min_dimension:
            scale = min_dimension / float(min_side)

        if scale != 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)
            return resized, scale
        return image, 1.0

    def remove_noise(self, gray_image: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """Applies gentle noise filtering without degrading small text character edges."""
        if method == "bilateral":
            # Preserves sharp edges while smoothing flat noise
            return cv2.bilateralFilter(gray_image, d=7, sigmaColor=50, sigmaSpace=50)
        elif method == "median":
            return cv2.medianBlur(gray_image, 3)
        else:
            return cv2.GaussianBlur(gray_image, (3, 3), 0)

    def estimate_skew_angle(self, gray_image: np.ndarray) -> float:
        """
        Detects text orientation skew angle using minAreaRect on edge contours.
        Returns angle in degrees to rotate (counter-clockwise).
        """
        # Threshold to isolate foreground features
        thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Use morphological opening to eliminate small dots/salt noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        angles = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            angle = rect[2]
            if w < h:
                angle = angle + 90.0
            # Bound between -45 and 45
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            if abs(angle) > 0.2:
                angles.append(angle)

        if len(angles) == 0:
            return 0.0

        median_angle = float(np.median(angles))
        # Ignore negligible tilt
        if abs(median_angle) < 0.5 or abs(median_angle) > 35:
            return 0.0
        return median_angle

    def deskew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image to compensate for detected skew angle."""
        if abs(angle) < 0.2:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Fill background with white (255)
        border_val = (255, 255, 255) if len(image.shape) == 3 else 255
        deskewed = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
        return deskewed

    def binarize(self, gray_image: np.ndarray, method: str = "otsu") -> np.ndarray:
        """
        Converts grayscale image to binary high-contrast black/white for OCR engines.
        """
        if method == "adaptive":
            return cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
            )
        # Default Otsu thresholding
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete preprocessing pipeline:
        1. Resize to target dimension
        2. Convert to grayscale
        3. Remove noise
        4. Detect and correct skew angle
        5. Generate binarized image for OCR
        Returns the preprocessed image and a metadata dictionary.
        """
        metadata: Dict[str, Any] = {}
        orig_h, orig_w = image.shape[:2]
        metadata["original_size"] = (orig_w, orig_h)

        # 1. Resize
        resized, scale = self.resize_image(image)
        metadata["scale_factor"] = scale
        metadata["resized_size"] = (resized.shape[1], resized.shape[0])

        # 2. Grayscale
        gray = self.to_grayscale(resized)

        # 3. Deskew
        skew_angle = self.estimate_skew_angle(gray)
        metadata["skew_angle_deg"] = round(skew_angle, 2)
        if abs(skew_angle) >= 0.5:
            gray = self.deskew(gray, skew_angle)

        # 4. Noise reduction
        denoised = self.remove_noise(gray, method="gaussian")

        # 5. High-contrast binarization
        binary = self.binarize(denoised, method="otsu")

        return binary, metadata
