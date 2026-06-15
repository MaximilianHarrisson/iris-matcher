from typing import Dict, List, Tuple

import numpy as np

from .iris_dataset import IrisDataset
from .iris_matcher import IrisMatcher


class Evaluator:
    def __init__(self, matcher: IrisMatcher, probes: IrisDataset) -> None:
        self.matcher = matcher
        self.probes = probes

    def evaluate(self, max_distance: int, match_on: str = 'identity',
                 ranks: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
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
        rank_hits = {k: 0 for k in ranks}

        def match_key(entry) -> str:
            return entry.person_id if match_on == 'person' else entry.identity

        for probe in probe_entries:
            try:
                ranked = self.matcher.ranked_candidates(probe.path, max_distance)
            except Exception as e:
                print(f'Error ({probe.path}): {e}')
                continue

            evaluated += 1
            penetration_rates.append(len(ranked) / gallery_size)

            probe_key = match_key(probe)
            keys = [key for key, _ in ranked]
            genuine_found = any(match_key(self.matcher.entries[c]) == probe_key for c in keys)
            if not genuine_found:
                errors += 1

            for k in ranks:
                if any(match_key(self.matcher.entries[c]) == probe_key for c in keys[:k]):
                    rank_hits[k] += 1

        if evaluated == 0:
            raise RuntimeError("No probes evaluated.")

        er = errors / evaluated
        hr = 1.0 - er
        pr = float(np.mean(penetration_rates)) if penetration_rates else 0.0
        rank_rates = {k: rank_hits[k] / evaluated for k in ranks}

        self.print_metrics(er, hr, pr, evaluated, gallery_size, rank_rates)

        result = {"er": er, "hr": hr, "pr": pr}
        for k in ranks:
            result[f"rank{k}"] = rank_rates[k]
        return result

    @staticmethod
    def print_metrics(er: float, hr: float, pr: float, n_probes: int, gallery_size: int,
                      rank_rates: Dict[int, float] = None) -> None:
        print(f"Probes evaluated:      {n_probes}")
        print(f"Gallery size:          {gallery_size}")
        print(f"Error Rate (ER):       {er * 100:.2f}%")
        print(f"Hit Rate   (HR):       {hr * 100:.2f}%")
        print(f"Penetration Rate (PR): {pr * 100:.4f}%")
        if rank_rates:
            ranks_str = '  '.join(f'rank-{k}: {v * 100:.2f}%' for k, v in sorted(rank_rates.items()))
            print(f"CMC:                   {ranks_str}")
