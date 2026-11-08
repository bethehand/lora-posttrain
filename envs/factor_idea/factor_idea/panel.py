"""Daily panels. dev.npz stops at 2025-12-31; the sealed test has to be asked for by name."""
from __future__ import annotations

import datetime as dt
import os
import sys
from dataclasses import dataclass

import numpy as np

from factor_idea.dsl import FIELDS

DATA_DIR = os.environ.get("FACTOR_IDEA_DATA", "/mnt/nvme1/bruce/data/factor_idea_v1")
SEALED_CONFIRMATION = "open the sealed test"


@dataclass(frozen=True)
class Panel:
    dates: np.ndarray  # datetime64[D], the close time of each row's bar
    sec_ids: np.ndarray
    symbols: np.ndarray
    fields: dict[str, np.ndarray]
    universe: np.ndarray  # T x N bool

    def row(self, date: str, *, side: str = "left") -> int:
        return int(np.searchsorted(self.dates, np.datetime64(date), side=side))

    def upto(self, last: str) -> "Panel":
        """The same panel without any row after ``last``."""
        end = self.row(last, side="right")
        return Panel(
            self.dates[:end],
            self.sec_ids,
            self.symbols,
            {name: a[:end] for name, a in self.fields.items()},
            self.universe[:end],
        )


def _load(path: str) -> Panel:
    with np.load(path, allow_pickle=False) as z:
        return Panel(
            z["dates"].astype("datetime64[D]"),
            z["sec_ids"],
            z["symbols"],
            {name: z[name] for name in FIELDS},
            z["universe"],
        )


def load_dev(data_dir: str | None = None) -> Panel:
    return _load(os.path.join(data_dir or DATA_DIR, "dev.npz"))


def load_sealed(confirm: str, data_dir: str | None = None) -> Panel:
    """Every row, 2026 included. Each opening is written to sealed_access.log."""
    if confirm != SEALED_CONFIRMATION:
        raise PermissionError(f"pass confirm={SEALED_CONFIRMATION!r} to read the sealed test period")
    root = data_dir or DATA_DIR
    with open(os.path.join(root, "sealed_access.log"), "a") as log:
        log.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {' '.join(sys.argv)}\n")
    return _load(os.path.join(root, "sealed.npz"))
