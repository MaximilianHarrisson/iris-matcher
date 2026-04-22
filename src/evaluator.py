from .iris_entry import *
from .iris_matcher import *


class Evaluator:
    def __init__(self, matcher: IrisMatcher) -> None:
        self.matcher = matcher

    def evaluate(self, max_distance: int) -> Dict[str, float]:
        mih = self.matcher.mih
        paths = list(mih.dataset.keys())
        total = len(paths)

        if total < 2:
            raise ValueError("Dataset length should be greater than 1 to evaluate.")

        errors = 0
        penetration_rates: List[float] = []

        for query_path in paths:
            query_vec = mih.dataset[query_path]
            query_id = parse_entry(query_path).person_id

            candidates: set = mih.collect_candidates(query_vec, max_distance)
            candidates.discard(query_path)

            penetration_rates.append(len(candidates) / (total - 1))

            genuine_found = any(parse_entry(candidate).person_id == query_id for candidate in candidates)
            if not genuine_found:
                errors += 1

        er = errors / total
        hr = 1.0 - er
        pr = float(np.mean(penetration_rates)) if penetration_rates else 0.0

        self.print_metrics(er, hr, pr)
        return {"er": er, "hr": hr, "pr": pr}

    @staticmethod
    def print_metrics(er: float, hr: float, pr: float) -> None:
        print(f"Error Rate (ER):       {er * 100:.2f}%")
        print(f"Hit Rate   (HR):       {hr * 100:.2f}%")
        print(f"Penetration Rate (PR): {pr * 100:.4f}%")
