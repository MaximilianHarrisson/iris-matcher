from typing import Optional

import numpy as np
import torch


class ThermometerBinarizer:
    def __init__(self, levels: int = 8):
        if levels < 1:
            raise ValueError('levels must be >= 1')
        self.levels = levels

    def binarize(self, features: np.ndarray) -> np.ndarray:
        flat = np.asarray(features, dtype=np.float32).reshape(-1)
        x = torch.from_numpy(flat)

        lower = float(x.min().item())
        upper = float(x.max().item())

        if upper <= lower:
            return np.zeros(flat.size * self.levels, dtype=np.uint8)

        thresholds = torch.linspace(lower, upper, self.levels + 2, dtype=torch.float32)[1:-1]
        encoded = (x.unsqueeze(1) >= thresholds.unsqueeze(0)).to(torch.uint8)
        return encoded.flatten().numpy()
