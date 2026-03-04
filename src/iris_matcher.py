import os
import numpy as np
import bitarray

from .feature_extractor import FeatureExtractor
from .thermometer import Thermometer
from .mih import MIH


class IrisMatcher:
    def __init__(self, casia_path: str, num_tables: int = 4):
        self.casia_path = casia_path
        self.extractor = FeatureExtractor()
        self.encoder = Thermometer()
        self.mih = MIH(num_tables=num_tables)

    def build_index(self):
        vectors = {}
        max_len = 0
        descriptors_list = []

        for root, _, files in os.walk(self.casia_path):
            for f in files:
                if f.lower().endswith('.jpg'):
                    path = os.path.join(root, f)
                    try:
                        des = self.extractor.extract(path)
                        if des is not None:
                            descriptors_list.append(des)
                    except Exception as e:
                        print(f'Error ({path}): {e}')

        self.encoder.fit(descriptors_list)

        for root, _, files in os.walk(self.casia_path):
            for f in files:
                if f.lower().endswith('.jpg'):
                    path = os.path.join(root, f)
                    try:
                        des = self.extractor.extract(path)
                        vec = self.encoder.to_bitarray(des)
                        vectors[path] = vec
                        if len(vec) > max_len:
                            max_len = len(vec)
                    except Exception as e:
                        print(f'Error ({path}): {e}')

        self.mih.max_len = max_len

        for path, vec in vectors.items():
            if len(vec) < max_len:
                padded = np.zeros(max_len, dtype=np.uint8)
                padded[:len(vec)] = vec
                vec = padded
            elif len(vec) > max_len:
                vec = vec[:max_len]
            self.mih.insert(path, bitarray.bitarray(vec.tolist()))

    def search(self, query_img: str, max_distance: int | None = None, tolerance_pct: float | None = None):
        des = self.extractor.extract(query_img)
        vec = self.encoder.to_bitarray(des)

        if len(vec) < self.mih.max_len:
            padded = np.zeros(self.mih.max_len, dtype=np.uint8)
            padded[:len(vec)] = vec
            vec = padded
        elif len(vec) > self.mih.max_len:
            vec = vec[:self.mih.max_len]

        if tolerance_pct is not None:
            max_distance = int(self.mih.max_len * tolerance_pct)

        return self.mih.query(bitarray.bitarray(vec.tolist()), max_distance=max_distance)
