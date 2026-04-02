"""
perception/preprocessor.py

OpenCV-based image preprocessing pipeline.

Steps applied (configurable):
  1. Gaussian denoising – reduces sensor noise before inference.
  2. CLAHE (Contrast-Limited Adaptive Histogram Equalization) – improves
     visibility of cracks and stains under varying lighting.
  3. Laplacian edge enhancement – optional, accentuates structural boundaries.
"""

from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocess raw frames before feeding them to the YOLO detector.

    Parameters
    ----------
    denoise : bool
        Apply Gaussian blur to suppress sensor noise.
    clahe : bool
        Apply CLAHE on the luminance channel to normalise exposure.
    edge_enhancement : bool
        Blend a Laplacian edge map into the image for crack accentuation.
    """

    def __init__(
        self,
        denoise: bool = True,
        clahe: bool = True,
        edge_enhancement: bool = False,
    ) -> None:
        self.denoise = denoise
        self.clahe = clahe
        self.edge_enhancement = edge_enhancement

        # Pre-build CLAHE object (reuse across frames for efficiency)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the full preprocessing pipeline to a BGR image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (as returned by ``cv2.imread``).

        Returns
        -------
        np.ndarray
            Preprocessed image in BGR format, ready for YOLO inference.
        """
        if image is None or image.size == 0:
            raise ValueError("ImagePreprocessor.process() received an empty image.")

        img = image.copy()

        if self.denoise:
            img = self._apply_denoise(img)

        if self.clahe:
            img = self._apply_clahe(img)

        if self.edge_enhancement:
            img = self._apply_edge_enhancement(img)

        return img

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_denoise(image: np.ndarray) -> np.ndarray:
        """Gaussian blur to suppress high-frequency noise."""
        return cv2.GaussianBlur(image, (3, 3), 0)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel of the LAB colour space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self._clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_edge_enhancement(image: np.ndarray) -> np.ndarray:
        """Blend a Laplacian edge map to enhance structural boundaries."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
        edge_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 0.8, edge_3ch, 0.2, 0)
