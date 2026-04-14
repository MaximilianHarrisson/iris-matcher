import os

from .iris_entry import *


class IrisDataset:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[IrisEntry]:
        entries = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith('.jpg'):
                    full_path = os.path.join(root, file)
                    entry = parse_entry(full_path)
                    if entry is not None:
                        entries.append(entry)
        return entries
