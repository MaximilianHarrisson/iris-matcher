import cv2
import numpy as np


class SiftExtractor:
    def __init__(self, n_features: int = 2000):
        self.sift = cv2.SIFT.create(n_features)

    def extract(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'File not found: {img_path}')

        _, descriptors = self.sift.detectAndCompute(img, None)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(128, dtype=np.float32)

        return descriptors.mean(axis=0).astype(np.float32)
