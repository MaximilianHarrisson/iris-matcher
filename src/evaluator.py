from typing import Dict, List

import numpy as np

from .iris_dataset import IrisDataset
from .iris_entry import parse_entry
from .iris_matcher import IrisMatcher


class Evaluator:
    def __init__(self, matcher: IrisMatcher, probes: IrisDataset) -> None:
        self.matcher = matcher
        self.probes = probes

    def evaluate(self, max_distance: int) -> Dict[str, float]:
        gallery = self.matcher.mih.dataset
        if not gallery:
            raise ValueError("Gallery is empty. Build the index before evaluating.")

        probe_entries = self.probes.load()
        if not probe_entries:
            raise ValueError("Empty probes dataset.")

        gallery_size = len(gallery)
        errors = 0
        evaluated = 0
        penetration_rates: List[float] = []

        for probe in probe_entries:
            try:
                candidates = self.matcher.candidates(probe.path, max_distance)
            except Exception as e:
                print(f'Error ({probe.path}): {e}')
                continue

            evaluated += 1
            penetration_rates.append(len(candidates) / gallery_size)

            genuine_found = any(parse_entry(c).person_id == probe.person_id for c in candidates)
            if not genuine_found:
                errors += 1

        if evaluated == 0:
            raise RuntimeError("No probes evaluated.")

        er = errors / evaluated
        hr = 1.0 - er
        pr = float(np.mean(penetration_rates)) if penetration_rates else 0.0

        self.print_metrics(er, hr, pr, evaluated, gallery_size)
        return {"er": er, "hr": hr, "pr": pr}

    @staticmethod
    def print_metrics(er: float, hr: float, pr: float, n_probes: int, gallery_size: int) -> None:
        print(f"Probes evaluated:      {n_probes}")
        print(f"Gallery size:          {gallery_size}")
        print(f"Error Rate (ER):       {er * 100:.2f}%")
        print(f"Hit Rate   (HR):       {hr * 100:.2f}%")
        print(f"Penetration Rate (PR): {pr * 100:.4f}%")
