import cv2
import numpy as np


class OrbExtractor:
    def __init__(self, n_features: int = 2000):
        self.orb = cv2.ORB.create(n_features)

    def extract(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'File not found: {img_path}')

        _, descriptors = self.orb.detectAndCompute(img, None)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(256, dtype=np.uint8)

        bits = np.unpackbits(descriptors, axis=1)
        return (bits.mean(axis=0) >= 0.5).astype(np.uint8)
