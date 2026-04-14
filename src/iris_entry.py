from dataclasses import *
from pathlib import *
from typing import *


@dataclass
class IrisEntry:
    path: str
    person_id: str
    side: str
    name: str

    @property
    def identity(self) -> str:
        return f'{self.person_id}_{self.side}'


def parse_entry(path: str) -> Optional[IrisEntry]:
    try:
        p = Path(path)
        parts = p.parts
        return IrisEntry(
            path=path,
            person_id=parts[-3],
            side=parts[-2],
            name=p.stem,
        )
    except IndexError:
        return None
