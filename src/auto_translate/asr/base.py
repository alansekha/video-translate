from typing import Protocol

import numpy as np


class ASR(Protocol):
    """Any class with this method counts as an ASR backend (like a Go interface)."""

    def transcribe(self, audio: np.ndarray, probe: bool = False) -> str:
        """audio: float32 mono 16 kHz. Returns Japanese text, or "" if nothing usable.

        probe: the VAD heard no speech here, so only keep confident results."""
        ...
