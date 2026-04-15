from typing import *

import bitarray
import bitarray.util


class MIH:
    def __init__(self, num_tables: int = 4):
        self.num_tables = num_tables
        self.hash_tables: List[Dict[str, List[str]]] = []
        self.dataset: Dict[str, bitarray.bitarray] = {}
        self.segment_size = None
        self.max_len = 0

    def build_tables(self, bit_length: int) -> None:
        self.segment_size = bit_length // self.num_tables
        self.hash_tables = [{} for _ in range(self.num_tables)]

    def insert(self, key: str, vec: bitarray.bitarray) -> None:
        if self.segment_size is None:
            self.build_tables(len(vec))

        self.dataset[key] = vec
        for i in range(self.num_tables):
            seg = vec[i * self.segment_size:(i + 1) * self.segment_size].to01()
            if seg not in self.hash_tables[i]:
                self.hash_tables[i][seg] = []
            self.hash_tables[i][seg].append(key)

    def hamming_distance(self, a: bitarray.bitarray, b: bitarray.bitarray) -> int:
        return bitarray.util.count_xor(a, b)

    def generate_hamming_ball(self, segment: str, radius: int) -> List[str]:
        from itertools import combinations

        indexes = range(len(segment))
        hamming_ball = [segment]

        for r in range(1, radius + 1):
            for positions in combinations(indexes, r):
                flipped = list(segment)
                for p in positions:
                    flipped[p] = '1' if flipped[p] == '0' else '0'
                hamming_ball.append(''.join(flipped))
        return hamming_ball

    def collect_candidates(self, vec: bitarray.bitarray, max_distance: int) -> Set[str]:
        segment_radius = max_distance // self.num_tables
        selected_keys = set()
        for i in range(self.num_tables):
            segment = vec[i * self.segment_size:(i + 1) * self.segment_size].to01()
            candidates = self.generate_hamming_ball(segment, segment_radius)
            for candidate in candidates:
                if candidate in self.hash_tables[i]:
                    for key in self.hash_tables[i][candidate]:
                        selected_keys.add(key)
        return selected_keys

    def query(self, vec: bitarray.bitarray, max_distance: int = 8) -> List[Tuple[str, int, str]]:
        selected_keys = self.collect_candidates(vec, max_distance)

        results = []
        for key in selected_keys:
            value = self.dataset[key]
            distance = self.hamming_distance(vec, value)
            results.append((key, distance, value.to01()))
        return sorted(results, key=lambda x: x[1])
