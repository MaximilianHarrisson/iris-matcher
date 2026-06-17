"""
Stratified train/test split for the iris dataset.

Each (person_id, side) pair is shuffled independently with a fixed seed and
partitioned by the train ratio, guaranteeing every identity has images in both
splits. Output goes to <out>/train/<person_id>/<side>/ and <out>/test/...

Default is copy mode (safer). Use --mode move to free disk space — the source
directory can be re-extracted from the CASIA zip if needed.
"""

import argparse
import random
import shutil
from pathlib import Path


def collect_identities(source: Path) -> dict:
    identities = {}
    for person_dir in sorted(source.iterdir()):
        if not person_dir.is_dir() or person_dir.name in ('train', 'test'):
            continue
        for side_dir in sorted(person_dir.iterdir()):
            if not side_dir.is_dir():
                continue
            images = sorted(p for p in side_dir.iterdir() if p.suffix.lower() == '.jpg')
            if images:
                identities[(person_dir.name, side_dir.name)] = images
    return identities


def split_identity(images: list, train_ratio: float, rng: random.Random) -> tuple:
    shuffled = images[:]
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_ratio))
    n_train = max(1, min(n_train, len(shuffled) - 1))
    return shuffled[:n_train], shuffled[n_train:]


def transfer(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'copy':
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser(description='Stratified train/test split for iris dataset.')
    parser.add_argument('--source', default='../assets/iris', help='Root with <person_id>/<side>/*.jpg')
    parser.add_argument('--out', default='../assets/iris', help='Output root; creates train/ and test/ inside')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=95)
    parser.add_argument('--mode', choices=['copy', 'move'], default='copy')
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    train_dir = out / 'train'
    test_dir = out / 'test'

    if not source.exists():
        raise FileNotFoundError(f'Source not found: {source}')

    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    rng = random.Random(args.seed)
    identities = collect_identities(source)

    if not identities:
        raise RuntimeError(f'No identities found under {source}.')

    train_count = test_count = 0
    sizes = []
    for (person_id, side), images in identities.items():
        train_imgs, test_imgs = split_identity(images, args.train_ratio, rng)
        sizes.append((len(train_imgs), len(test_imgs)))

        for img in train_imgs:
            transfer(img, train_dir / person_id / side / img.name, args.mode)
            train_count += 1
        for img in test_imgs:
            transfer(img, test_dir / person_id / side / img.name, args.mode)
            test_count += 1

    train_per_id = train_count / len(identities)
    test_per_id = test_count / len(identities)
    min_train = min(s[0] for s in sizes)
    min_test = min(s[1] for s in sizes)

    print(f'Identities:      {len(identities)}')
    print(f'Train images:    {train_count}  ({train_per_id:.2f} avg, {min_train} min per identity)')
    print(f'Test images:     {test_count}  ({test_per_id:.2f} avg, {min_test} min per identity)')
    print(f'Mode:            {args.mode}')
    print(f'Seed:            {args.seed}')
    print(f'Train dir:       {train_dir}')
    print(f'Test dir:        {test_dir}')


if __name__ == '__main__':
    main()
