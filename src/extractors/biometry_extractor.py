from typing import *

import cv2
import numpy as np


STRIP_HEIGHT = 8
STRIP_WIDTH = 128

GABOR_PARAMS = [
    {'frequency': 0.1, 'theta': 0},
    {'frequency': 0.1, 'theta': np.pi / 2},
]

MIN_KEYPOINTS = 20


def _segment(img: np.ndarray) -> Optional[tuple]:
    blurred = cv2.GaussianBlur(img, (7, 7), 0)

    iris_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=100,
        param1=50,
        param2=20,
        minRadius=60,
        maxRadius=200,
    )
    if iris_circles is None:
        return None
    iris = iris_circles[0][0].astype(int)

    pupil_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=80,
        param2=30,
        minRadius=20,
        maxRadius=int(iris[2] * 0.65),
    )
    if pupil_circles is None:
        return None
    pupil = pupil_circles[0][0].astype(int)

    return iris, pupil


def _normalize(img: np.ndarray, iris: np.ndarray, pupil: np.ndarray) -> np.ndarray:
    strip = np.zeros((STRIP_HEIGHT, STRIP_WIDTH), dtype=np.float32)
    h, w = img.shape

    for j in range(STRIP_WIDTH):
        theta = 2 * np.pi * j / STRIP_WIDTH
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        for i in range(STRIP_HEIGHT):
            r = i / STRIP_HEIGHT

            x = int((1 - r) * (pupil[0] + pupil[2] * cos_t) + r * (iris[0] + iris[2] * cos_t))
            y = int((1 - r) * (pupil[1] + pupil[2] * sin_t) + r * (iris[1] + iris[2] * sin_t))

            if 0 <= x < w and 0 <= y < h:
                strip[i, j] = img[y, x]

    return strip


def _gabor_iris_code(strip: np.ndarray) -> np.ndarray:
    bits = []
    for params in GABOR_PARAMS:
        kernel = cv2.getGaborKernel(
            ksize=(31, 31),
            sigma=4.0,
            theta=params['theta'],
            lambd=1.0 / params['frequency'],
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F,
        )
        response = cv2.filter2D(strip, cv2.CV_32F, kernel)
        bits.append((response >= 0).astype(np.uint8).flatten())

    return np.concatenate(bits)


class BiometryExtractor:
    def extract(self, img_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'File not found: {img_path}')

        result = _segment(img)
        if result is None:
            return None
        iris, pupil = result

        strip = _normalize(img, iris, pupil)
        return _gabor_iris_code(strip)
