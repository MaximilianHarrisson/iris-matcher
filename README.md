# Evaluation of Feature Extractors for Iris Recognition with Multi-Index Hashing

This project reproduces the experiments from the TCC article *"Avaliação de Extratores de Características para Reconhecimento de Íris com Multi-Index Hashing"* (UFF).

Seven extractors are compared: VGG16, ResNet50, ConvNeXt-Tiny, MobileNetV2, InceptionV3, SIFT, and ORB. The metrics used on the CASIA-Iris-Lamp dataset are hit rate, penetration rate, and rank-k.

## Requirements

- Python 3.8

```bash
pip install -r requirements.txt
```

## Dataset

The experiments use [CASIA-Iris-Lamp](https://hycasia.github.io/dataset/casia-irisv4/), a subset of CASIA-IrisV4 with 16212 images from 411 subjects. After downloading and extracting, run the stratified split to create train and test sets:

```bash
python scripts/split_dataset.py --source <path-to-casia-lamp> --out assets/iris
```

This creates `assets/iris/train/` and `assets/iris/test/` with every identity represented in both splits (80/20 ratio, fixed seed).

## Running the experiments

Feature vectors are cached on first extraction to `results/cache/` — subsequent runs reuse them without re-extracting.

### Experiment 1 — Extractor comparison

Evaluates all 7 extractors across two identification scenarios (person-level and person+side) with 32-bit MIH segments and per-segment Hamming radius 2 (`m = B/32` tables per extractor).

```bash
python scripts/collect_results.py [run-label]
```

Each run is tagged with the given label, or a timestamp when omitted. Outputs `results/quality_<run-tag>.csv`.

### Experiment 2 — Number of tables sweep

Included in the same `collect_results.py` run. Varies the number of MIH tables for VGG16 (`m ∈ {4, 8, 16, 32}`) to show the trade-off between hit rate and penetration rate. Outputs `results/tables_<run-tag>.csv`.

### Experiment 3 — Thermometer binarization

Compares thermometer encoding (levels `k ∈ {8, 10, 12, 14, 16, 18, 20}`, `m = B/32` tables) against median binarization for VGG16.

```bash
python scripts/thermometer_sweep.py
```

Outputs `results/thermometer_scaled.csv` and per-level logs at `results/collect/thermometer_scaled/`.

## Project structure

```
main.py        Builds the default index (VGG16 + median) and runs a single query
src/           Extractors, binarizers, MIH, IrisMatcher, Evaluator
scripts/       Experiment and utility scripts
assets/
  iris/        Dataset split (train/ and test/)
results/
  cache/       Cached feature vectors (.npz), reused across runs
  collect/     Per-run logs (collect_results.py and thermometer_sweep.py)
```
