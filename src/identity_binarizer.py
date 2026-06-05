import numpy as np


class IdentityBinarizer:
    def binarize(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features).reshape(-1).astype(np.uint8)
