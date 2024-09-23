from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass
class ReplayBuffer(Generic[T]):
    capacity: int
    seed: int | None = None
    _items: deque[T] = field(init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items = deque(maxlen=self.capacity)
        self._rng = random.Random(self.seed)

    def add_many(self, items: Iterable[T]) -> None:
        for item in items:
            self._items.append(item)

    def sample(self, count: int) -> list[T]:
        if count <= 0:
            raise ValueError("count must be positive")
        available = list(self._items)
        if count >= len(available):
            return available
        return self._rng.sample(available, count)

    def mix_with_current(self, current: list[T], replay_count: int) -> list[T]:
        mixed = list(current)
        mixed.extend(self.sample(replay_count) if self._items else [])
        self.add_many(current)
        return mixed

    def __len__(self) -> int:
        return len(self._items)

