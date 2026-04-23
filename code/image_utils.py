"""Image preprocessing helpers shared by training and inference code."""

import cv2
import numpy as np
from monai.transforms import MapTransform


class HistogramEqualizationd(MapTransform):
    """Apply histogram equalization to MONAI dictionary items.

    Two modes are supported:
    - 2D arrays use standard histogram equalization,
    - channel-first image tensors use CLAHE independently per channel.
    """

    def __init__(self, keys):
        """Store the dictionary keys that should be equalized."""
        super().__init__(keys)

    def __call__(self, data):
        """Equalize all configured keys and return a copied data dictionary."""
        d = dict(data)
        for key in self.keys:
            img = d[key]

            # Use classic histogram equalization for single-channel images.
            if img.ndim == 2:
                d[key] = cv2.equalizeHist(img.astype(np.uint8))
                continue

            # Use CLAHE per channel for RGB-style channel-first tensors.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            d[key] = np.stack([clahe.apply(img[i].astype(np.uint8)) for i in range(img.shape[0])])
        return d
