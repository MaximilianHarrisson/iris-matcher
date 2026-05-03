import os
from typing import *

import numpy as np
import bitarray

from .thermometer_binarizer import ThermometerBinarizer
from .iris_dataset import IrisDataset
from .iris_entry import IrisEntry
from .median_binarizer import MedianBinarizer
from .mih import MIH
from .vgg16_extractor import Vgg16Extractor


class IrisMatcher:
    def __init__(self, dataset: IrisDataset, num_tables: int = 4):
        self.dataset = dataset
        self.extractor = Vgg16Extractor()
        self.binarizer = MedianBinarizer()
        self.mih = MIH(num_tables=num_tables)
        self.entries: Dict[str, IrisEntry] = {}
        self.indexes_path = os.path.join('results', 'indexes.npz')

    def _binarize(self, vec: np.ndarray) -> np.ndarray:
        return self.binarizer.binarize(vec)

    def _reset_mih(self) -> None:
        self.mih = MIH(num_tables=self.mih.num_tables)

    def _load_cached_vectors(self) -> Optional[Dict[str, np.ndarray]]:
        if not os.path.exists(self.indexes_path):
            return None

        try:
            data = np.load(self.indexes_path, allow_pickle=False)
            paths = data['paths']
            vectors = data['vectors']
            max_len = int(data['max_len'])

            if paths.ndim != 1 or vectors.ndim != 2 or len(paths) != vectors.shape[0]:
                raise ValueError('Invalid cache structure.')
            if vectors.shape[1] != max_len:
                raise ValueError('Invalid max_len in cache.')

            result: Dict[str, np.ndarray] = {}
            for i, path in enumerate(paths.tolist()):
                result[str(path)] = vectors[i]
            return result
        except Exception as e:
            print(f'Failed to load cached indexes ({self.indexes_path}): {e}')
            return None

    def _save_cached_vectors(self, vectors: Dict[str, np.ndarray]) -> None:
        if not vectors:
            return

        os.makedirs(os.path.dirname(self.indexes_path), exist_ok=True)
        max_len = len(max(vectors.values(), key=len))
        paths = list(vectors.keys())
        dtype = next(iter(vectors.values())).dtype
        matrix = np.zeros((len(paths), max_len), dtype=dtype)

        for i, path in enumerate(paths):
            vec = vectors[path].reshape(-1)
            size = min(len(vec), max_len)
            matrix[i, :size] = vec[:size]

        np.savez_compressed(
            self.indexes_path,
            paths=np.array(paths),
            vectors=matrix,
            max_len=np.array(max_len, dtype=np.int32),
        )

    def _build_mih_from_vectors(self, vectors: Dict[str, np.ndarray]) -> None:
        if not vectors:
            return

        self._reset_mih()
        binary_vectors = {path: self._binarize(vec) for path, vec in vectors.items()}
        self.mih.max_len = len(max(binary_vectors.values(), key=len))

        for path, vec in binary_vectors.items():
            if len(vec) < self.mih.max_len:
                padded = np.zeros(self.mih.max_len, dtype=np.uint8)
                padded[:len(vec)] = vec
                vec = padded
            elif len(vec) > self.mih.max_len:
                vec = vec[:self.mih.max_len]
            self.mih.insert(path, bitarray.bitarray(vec.tolist()))

    def build_index(self) -> None:
        dataset_entries = self.dataset.load()
        self.entries = {entry.path: entry for entry in dataset_entries}
        dataset_paths = set(self.entries.keys())

        cached_vectors = self._load_cached_vectors()
        if cached_vectors is not None:
            vectors = {path: vec for path, vec in cached_vectors.items() if path in dataset_paths}
            if vectors:
                self._build_mih_from_vectors(vectors)
                print(f'Loaded {len(vectors)} vectors from cache: {self.indexes_path}')
                return
            print(f'Cached indexes found but no matching dataset paths. Rebuilding cache...')

        vectors = {}

        for entry in dataset_entries:
            try:
                vec = self.extractor.extract(entry.path)
                if vec is not None:
                    vectors[entry.path] = vec
            except Exception as e:
                print(f'Error ({entry.path}): {e}')

        if not vectors:
            print('No vectors were extracted. Index was not built.')
            return

        self._save_cached_vectors(vectors)
        self._build_mih_from_vectors(vectors)
        print(f'Generated and saved {len(vectors)} vectors to: {self.indexes_path}')

    def search(self, query_img: str, max_distance: Optional[int] = None, tolerance_pct: Optional[float] = None) -> List[Tuple[str, int, str]]:
        vec = self._binarize(self.extractor.extract(query_img))

        if tolerance_pct is not None:
            max_distance = int(self.mih.max_len * tolerance_pct)

        return self.mih.query(bitarray.bitarray(vec.tolist()), max_distance=max_distance)

    def candidates(self, query_img: str, max_distance: int) -> Set[str]:
        vec = self._binarize(self.extractor.extract(query_img))
        return self.mih.collect_candidates(bitarray.bitarray(vec.tolist()), max_distance)
