import math
import numpy as np


class Thermometer:
    def __init__(self, num_bits: int = 8):
        self.num_bits: int = num_bits
        self.min_value: int | None = None
        self.max_value: int | None = None

    def fit(self, descriptors: list[np.ndarray]) -> None:
        all_values = np.concatenate([d.flatten() for d in descriptors if d is not None])
        self.min_value = int(np.min(all_values))
        self.max_value = int(np.max(all_values))
        print(f'Thermometer range: [{self.min_value}, {self.max_value}]')

    def encode_value(self, data: float) -> np.ndarray:
        bits = np.zeros(self.num_bits, dtype=np.uint8)
        bits_activated = int(math.ceil(((data - self.min_value) / (self.max_value - self.min_value)) * self.num_bits))
        bits[:bits_activated] = 1
        return bits

    def to_bitarray(self, descriptors: np.ndarray | None) -> np.ndarray:
        if descriptors is None:
            return np.array([], dtype=np.uint8)

        encoded = [self.encode_value(v) for v in descriptors.flatten()]
        return np.concatenate(encoded)
