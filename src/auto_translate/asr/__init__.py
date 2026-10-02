from ..config import AsrConfig
from .base import ASR


def create_asr(cfg: AsrConfig) -> ASR:
    # Imports live inside the branches so unused backends (and their deps) are never loaded.
    if cfg.backend == "faster-whisper":
        from .faster_whisper import FasterWhisperASR

        return FasterWhisperASR(cfg)
    raise ValueError(f"unknown ASR backend: {cfg.backend!r}")
