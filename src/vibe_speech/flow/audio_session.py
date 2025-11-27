from __future__ import annotations

import numpy as np
from typing import Optional


class AudioSession:
    """Buffers audio chunks for a listening window."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def append(self, chunk: Optional[np.ndarray]) -> None:
        if chunk is not None:
            self._chunks.append(chunk)

    def consume(self) -> Optional[np.ndarray]:
        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks)
        self._chunks = []
        return audio

    def reset(self) -> None:
        self._chunks = []
