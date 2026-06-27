"""Thermometer binarization sweep with scaled tables (m = B/32).

Experiment: VGG16, scenario 1, thermometer levels in [8, 10, 12, 14, 16, 18, 20].
For each level L, the thermometer code has 512*L bits, so m = 512*L // 32 tables.
This equalizes per-table search difficulty with the median experiments.

Raw VGG16 feature vectors are reused from the existing cache (cache_key='vgg16'),
so no re-extraction is needed.
"""
import csv
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

TRAIN_PATH = './assets/iris/train/'
TEST_PATH  = './assets/iris/test/'
RESULTS_DIR = './results'
LOG_DIR = './results/collect/thermometer_scaled'
CELL_TIMEOUT_S = 3600

VGG16_DIM = 512
SEGMENT_BITS = 32
RADIUS = 2
LEVELS = [8, 10, 12, 14, 16, 18, 20]
SCENARIO = 'person'


def run_cell(levels, num_tables, log_path, result_queue):
    sys.path.insert(0, str(ROOT))
    import torch
    from src.iris_dataset import IrisDataset
    from src.iris_matcher import IrisMatcher
    from src.evaluator import Evaluator
    from src.extractors.vgg16_extractor import Vgg16Extractor
    from src.binarizers.thermometer_binarizer import ThermometerBinarizer

    log_file = open(log_path, 'w', buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = log_file
    try:
        print(f'levels={levels} num_tables={num_tables} cuda={torch.cuda.is_available()}')
        train_set = IrisDataset(TRAIN_PATH)
        test_set  = IrisDataset(TEST_PATH)
        matcher = IrisMatcher(
            train_set,
            num_tables=num_tables,
            extractor=Vgg16Extractor(),
            binarizer=ThermometerBinarizer(levels=levels),
            cache_key='vgg16',
        )
        matcher.build_index()
        matcher.precompute_probes(test_set)

        evaluator = Evaluator(matcher, test_set)
        max_distance = RADIUS * num_tables
        t = time.perf_counter()
        metrics = evaluator.evaluate(max_distance=max_distance, match_on=SCENARIO, ranks=(1, 5))
        elapsed = time.perf_counter() - t
        result_queue.put({
            'status': 'ok',
            'hr': metrics['hr'], 'pr': metrics['pr'],
            'rank1': metrics['rank1'], 'rank5': metrics['rank5'],
            'search_time_s': elapsed,
        })
    except Exception:
        traceback.print_exc()
        result_queue.put({'status': 'error'})
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log_file.close()


def run_one(levels, num_tables):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'thermo_L{levels}.log')
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=run_cell, args=(levels, num_tables, log_path, queue),
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
        return {'status': 'error'}


def main():
    rows = []
    header = ['levels', 'num_tables', 'code_bits', 'hr', 'pr', 'rank1', 'rank5', 'search_time_s']

    for L in LEVELS:
        num_tables = (VGG16_DIM * L) // SEGMENT_BITS
        code_bits = VGG16_DIM * L
        print(f'levels={L} -> code={code_bits} bits, {num_tables} tables ... ', end='', flush=True)
        res = run_one(L, num_tables)
        if res['status'] == 'ok':
            rows.append([L, num_tables, code_bits,
                         f"{res['hr']:.6f}", f"{res['pr']:.6f}",
                         f"{res['rank1']:.6f}", f"{res['rank5']:.6f}",
                         f"{res['search_time_s']:.1f}"])
            print(f"hr={res['hr']:.4f} pr={res['pr']:.4f} "
                  f"r1={res['rank1']:.4f} r5={res['rank5']:.4f} "
                  f"t={res['search_time_s']:.0f}s")
        else:
            rows.append([L, num_tables, code_bits, res['status'], '-', '-', '-', '-'])
            print(res['status'])

    out_path = os.path.join(RESULTS_DIR, 'thermometer_scaled.csv')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
