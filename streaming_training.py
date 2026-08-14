"""Bounded deterministic sampling primitives for commissioning training."""
from __future__ import annotations

from dataclasses import dataclass
import math
import zlib

import numpy as np
import pandas as pd


def machine_seed(base_seed: int, machine_id: str, purpose: str) -> int:
    """Return a stable per-machine seed independent of Python hash randomization."""
    token = f"{machine_id}\0{purpose}".encode("utf-8")
    return (int(base_seed) ^ zlib.crc32(token)) & 0xFFFFFFFF


class PriorityReservoir:
    """Uniform fixed-size reservoir using deterministic random priorities.

    Every offered row receives one independent priority. Keeping the smallest
    priorities is equivalent to classic reservoir sampling and remains stable
    when the database fetch chunk size changes.
    """

    _PRIORITY = "__training_reservoir_priority"

    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        if self.capacity < 0:
            raise ValueError("reservoir capacity cannot be negative")
        self._rng = np.random.default_rng(int(seed))
        self._frame: pd.DataFrame | None = None
        self.seen = 0

    def offer(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.seen += len(frame)
        if self.capacity == 0:
            return
        candidate = frame.copy()
        candidate[self._PRIORITY] = self._rng.random(len(candidate))
        if self._frame is not None and not self._frame.empty:
            candidate = pd.concat([self._frame, candidate], ignore_index=True)
        if len(candidate) > self.capacity:
            priorities = candidate[self._PRIORITY].to_numpy(dtype=float)
            selected = np.argpartition(priorities, self.capacity - 1)[:self.capacity]
            candidate = candidate.iloc[selected]
        self._frame = candidate.reset_index(drop=True)

    def result(self) -> pd.DataFrame:
        if self._frame is None:
            return pd.DataFrame()
        return self._frame.drop(columns=[self._PRIORITY]).reset_index(drop=True)


@dataclass
class ForwardHoldoutReservoir:
    """Bounded fit reservoir plus an exact newest-row validation tail."""

    capacity: int
    holdout_fraction: float
    minimum_holdout: int
    seed: int

    def __post_init__(self):
        self.capacity = int(self.capacity)
        if self.capacity < 1:
            raise ValueError("training capacity must be positive")
        self._tail_capacity = max(
            int(self.minimum_holdout),
            int(math.ceil(self.capacity * float(self.holdout_fraction))),
        )
        if self._tail_capacity >= self.capacity:
            raise ValueError(
                "training capacity must leave room for both model fitting and validation"
            )
        self._fit = PriorityReservoir(
            self.capacity - self._tail_capacity,
            self.seed,
        )
        self._tail: pd.DataFrame | None = None
        self.seen = 0

    def offer(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.seen += len(frame)
        combined = frame.copy() if self._tail is None else pd.concat(
            [self._tail, frame], ignore_index=True
        )
        excess = max(0, len(combined) - self._tail_capacity)
        if excess:
            self._fit.offer(combined.iloc[:excess].reset_index(drop=True))
        self._tail = combined.iloc[excess:].reset_index(drop=True)

    def result(self, timestamp_column: str) -> pd.DataFrame:
        fit = self._fit.result()
        parts = [part for part in (fit, self._tail) if part is not None and not part.empty]
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True).sort_values(
            timestamp_column
        ).reset_index(drop=True)

    @property
    def retained(self) -> int:
        fit_rows = len(self._fit.result())
        tail_rows = 0 if self._tail is None else len(self._tail)
        return fit_rows + tail_rows
