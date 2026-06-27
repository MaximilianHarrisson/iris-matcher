import cv2
import numpy as np


class FeatureExtractor:
    def __init__(self, n_features: int = 2000):
        self.sift = cv2.SIFT.create(n_features)
        self.orb = cv2.ORB.create(n_features)

    def extract(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'File not found: {img_path}')
        # _, des = self.sift.detectAndCompute(img, None)
        # return des
        _, descriptors = self.orb.detectAndCompute(img, None)
        bits = np.unpackbits(descriptors, axis=1)

        result = (bits.mean(axis=0) >= 0.5).astype(np.uint8)

        return result
