"""Collect TCC comparison results in the FEASIBLE regime, with HR / PR / CMC rank-k.

Regime (validated as feasible; swap when the parameter decision is final):
  - 1-bit MEDIAN binarization for the CNNs and SIFT; ORB uses its native binary code.
  - Fixed 32-bit segments: per extractor, num_tables = dim // 32 (Option A), so the per-table
    search difficulty is equalized across extractors of different code length.
  - Per-table Hamming radius 2, i.e. max_distance = 2 * num_tables.

Each cell builds the matcher (reusing the warmed raw-vector caches, cache_key=name, so there
is no re-extraction), then evaluates both scenarios, reporting hit rate (HR), penetration rate
(PR) and CMC identification rate at rank 1 and rank 5. Subprocess-isolated per cell with a
timeout, like the original sweep.

Experiments:
  - Quality:           7 extractors x 2 scenarios, fixed 32-bit segments.
  - Hash Table Impact: VGG16, scenario 1, num_tables in [4, 8, 16, 32].

The Thermometer Size Impact experiment is intentionally omitted here: at a fixed table count
the thermometer codes are too long for the bucket search (it explodes). It needs a regime
where the table count scales with the level count, which is pending the parameter decision.
"""
import csv
import multiprocessing
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

# Each run is tagged so its outputs land in their own files (kept for later comparison).
# Pass a label as argv[1], otherwise a timestamp is used.
RUN_TAG = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y%m%d_%H%M%S')

TRAIN_PATH = './assets/iris/train/'
TEST_PATH = './assets/iris/test/'
RESULTS_DIR = './results'
LOG_DIR = f'./results/collect/{RUN_TAG}'
CELL_TIMEOUT_S = 3600
SEGMENT_BITS = 32
RADIUS = 2
RANKS = (1, 5)

# Median-binarized code length per extractor (= feature dimension; ORB stays 256 native bits).
DIMS = {
    'sift': 128, 'orb': 256, 'vgg16': 512, 'convnext': 768,
    'mobilenetv2': 1280, 'resnet50': 2048, 'inceptionv3': 2048,
}
QUALITY_EXTRACTORS = ['sift', 'orb', 'vgg16', 'resnet50', 'convnext', 'mobilenetv2', 'inceptionv3']
SWEEP_EXTRACTOR = 'vgg16'
SWEEP_TABLES = [4, 8, 16, 32]
SWEEP_SCENARIO = 'person'
SCENARIOS = [('person', 'Scenario 1 (person)'), ('identity', 'Scenario 2 (person+side)')]


def make_extractor_and_binarizer(name):
    """Return (extractor, binarizer): median for CNNs/SIFT, identity for ORB's native bits."""
    from src.binarizers.median_binarizer import MedianBinarizer
    from src.binarizers.identity_binarizer import IdentityBinarizer
    from src.extractors.sift_extractor import SiftExtractor
    from src.extractors.orb_extractor import OrbExtractor
    from src.extractors.vgg16_extractor import Vgg16Extractor
    from src.extractors.resnet50_extractor import Resnet50Extractor
    from src.extractors.convnext_extractor import ConvNextExtractor
    from src.extractors.mobilenetv2_extractor import MobileNetV2Extractor
    from src.extractors.inceptionv3_extractor import InceptionV3Extractor

    cnns = {
        'vgg16': Vgg16Extractor,
        'resnet50': Resnet50Extractor,
        'convnext': ConvNextExtractor,
        'mobilenetv2': MobileNetV2Extractor,
        'inceptionv3': InceptionV3Extractor,
    }
    if name == 'sift':
        return SiftExtractor(), MedianBinarizer()
    if name == 'orb':
        return OrbExtractor(), IdentityBinarizer()
    if name in cnns:
        return cnns[name](), MedianBinarizer()
    raise ValueError(f'Unknown extractor: {name}')


def tables_for(name):
    """Fixed 32-bit segments: num_tables = dim // 32."""
    return max(1, DIMS[name] // SEGMENT_BITS)


def run_cell(name, num_tables, scenarios, log_path, result_queue):
    """Build one matcher (median, given num_tables), evaluate the requested scenarios."""
    sys.path.insert(0, str(ROOT))
    import torch
    from src.iris_dataset import IrisDataset
    from src.iris_matcher import IrisMatcher
    from src.evaluator import Evaluator

    log_file = open(log_path, 'w', buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = log_file
    try:
        print(f'cuda_available={torch.cuda.is_available()}')
        train_set = IrisDataset(TRAIN_PATH)
        test_set = IrisDataset(TEST_PATH)
        extractor, binarizer = make_extractor_and_binarizer(name)
        matcher = IrisMatcher(
            train_set, num_tables=num_tables,
            extractor=extractor, binarizer=binarizer, cache_key=name,
        )
        matcher.build_index()
        matcher.precompute_probes(test_set)

        evaluator = Evaluator(matcher, test_set)
        max_distance = RADIUS * num_tables
        out = {'status': 'ok', 'num_tables': num_tables, 'scenarios': {}}
        for match_on in scenarios:
            t = time.perf_counter()
            metrics = evaluator.evaluate(max_distance=max_distance, match_on=match_on, ranks=RANKS)
            out['scenarios'][match_on] = {
                'hr': metrics['hr'], 'pr': metrics['pr'],
                'rank1': metrics['rank1'], 'rank5': metrics['rank5'],
                'search_time_s': time.perf_counter() - t,
            }
        result_queue.put(out)
    except Exception as exc:
        traceback.print_exc()
        result_queue.put({'status': 'error', 'error': repr(exc)})
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log_file.close()


def run_one(name, num_tables, scenarios, tag):
    """Spawn one isolated, timeout-bounded cell. Returns the result dict."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'{tag}.log')
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=run_cell, args=(name, num_tables, scenarios, log_path, queue),
    )
    proc.start()
    proc.join(timeout=CELL_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return {'status': 'timeout'}
    try:
        return queue.get_nowait()
    except Exception:
        return {'status': 'error', 'error': 'no result returned'}


def experiment_quality():
    """7 extractors x 2 scenarios at fixed 32-bit segments -> results/quality.csv."""
    rows = []
    for name in QUALITY_EXTRACTORS:
        nt = tables_for(name)
        res = run_one(name, nt, [s for s, _ in SCENARIOS], f'quality_{name}')
        for match_on, label in SCENARIOS:
            if res.get('status') == 'ok':
                sc = res['scenarios'][match_on]
                rows.append([name, str(nt), label,
                             f"{sc['hr']:.6f}", f"{sc['pr']:.6f}",
                             f"{sc['rank1']:.6f}", f"{sc['rank5']:.6f}"])
                print(f"quality {name} {match_on}: hr={sc['hr']:.4f} pr={sc['pr']:.4f} "
                      f"r1={sc['rank1']:.4f} r5={sc['rank5']:.4f}")
            else:
                rows.append([name, str(nt), label, res.get('status', 'error'), '-', '-', '-'])
                print(f"quality {name} {match_on}: {res.get('status')}")
    _write_csv(f'quality_{RUN_TAG}.csv', ['extractor', 'num_tables', 'scenario', 'hr', 'pr', 'rank1', 'rank5'], rows)


def experiment_tables():
    """VGG16, scenario 1, varying hash tables -> results/tables.csv."""
    rows = []
    for nt in SWEEP_TABLES:
        res = run_one(SWEEP_EXTRACTOR, nt, [SWEEP_SCENARIO], f'tables_{nt}')
        if res.get('status') == 'ok':
            sc = res['scenarios'][SWEEP_SCENARIO]
            rows.append([str(nt), f"{sc['hr']:.6f}", f"{sc['pr']:.6f}",
                         f"{sc['rank1']:.6f}", f"{sc['rank5']:.6f}"])
            print(f"tables nt={nt}: hr={sc['hr']:.4f} pr={sc['pr']:.4f} "
                  f"r1={sc['rank1']:.4f} r5={sc['rank5']:.4f}")
        else:
            rows.append([str(nt), res.get('status', 'error'), '-', '-', '-'])
    _write_csv(f'tables_{RUN_TAG}.csv', ['num_tables', 'hr', 'pr', 'rank1', 'rank5'], rows)


def _pct(value):
    try:
        return f'{float(value) * 100:.2f}'
    except ValueError:
        return value


def _write_csv(filename, header, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'Wrote {filename}')



def main():
    print(f'Run tag: {RUN_TAG}')
    experiment_quality()
    experiment_tables()
    print('Done.')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
