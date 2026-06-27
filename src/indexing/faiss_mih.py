from typing import *

import numpy as np
import bitarray
import bitarray.util
import faiss


class FaissMIH:
    def __init__(self, num_tables=4):
        self.dataset: Dict[str, bitarray.bitarray] = {}
        self.max_len = 0
        self.num_tables = num_tables

        self._index: Optional[faiss.IndexBinaryFlat] = None
        self._keys: List[str] = []

    def build_tables(self, bit_length: int) -> None:
        self._index = faiss.IndexBinaryMultiHash(bit_length, self.num_tables, 8)
        self.max_len = bit_length

    def insert(self, key: str, vec: bitarray.bitarray) -> None:
        if self._index is None:
            self.build_tables(len(vec))

        self.dataset[key] = vec
        self._keys.append(key)

        packed = np.frombuffer(vec.tobytes(), dtype=np.uint8).reshape(1, -1)
        self._index.add(packed)

    def hamming_distance(self, a: bitarray.bitarray, b: bitarray.bitarray) -> int:
        return bitarray.util.count_xor(a, b)

    def collect_candidates(self, vec: bitarray.bitarray, max_distance: int) -> Set[str]:
        results = self.query(vec, max_distance)
        return {key for key, _, _ in results}

    def query(self, vec: bitarray.bitarray, max_distance: int = 8) -> List[Tuple[str, int, str]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        packed = np.frombuffer(vec.tobytes(), dtype=np.uint8).reshape(1, -1)

        k = min(self._index.ntotal, max(1, self._index.ntotal))
        distances, indices = self._index.search(packed, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if dist <= max_distance:
                key = self._keys[idx]
                value = self.dataset[key]
                results.append((key, int(dist), value.to01()))

        return sorted(results, key=lambda x: x[1])
