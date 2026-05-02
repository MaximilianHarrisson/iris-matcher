import numpy as np


class MedianBinarizer:
    def binarize(self, features: np.ndarray) -> np.ndarray:
        flat = features.reshape(-1)
        threshold = np.median(flat)
        return (flat >= threshold).astype(np.uint8)
