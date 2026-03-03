import bitarray
import bitarray.util


class MIH:
    def __init__(self, num_tables: int = 4):
        self.num_tables = num_tables
        self.hash_tables: list[dict[str, list[str]]] = []
        self.dataset: dict[str, bitarray.bitarray] = {}
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

    def query(self, vec: bitarray.bitarray, max_distance: int = 3) -> list[tuple[str, int]]:
        if len(vec) < self.max_len:
            pad = bitarray.bitarray(self.max_len - len(vec))
            pad.setall(0)
            vec = vec + pad
        elif len(vec) > self.max_len:
            vec = vec[:self.max_len]

        candidates = set()
        for i in range(self.num_tables):
            seg = vec[i*self.segment_size:(i+1)*self.segment_size].to01()
            if seg in self.hash_tables[i]:
                candidates.update(self.hash_tables[i][seg])

        results = []
        for key in candidates:
            dist = self.hamming_distance(vec, self.dataset[key])
            if dist <= max_distance:
                results.append((key, dist))
        return sorted(results, key=lambda x: x[1])
