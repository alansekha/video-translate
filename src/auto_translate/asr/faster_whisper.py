import logging

import numpy as np

from .. import cuda_dlls
from ..config import AsrConfig

cuda_dlls.setup()  # must run before faster_whisper loads the CUDA libraries

from faster_whisper import WhisperModel  # noqa: E402

log = logging.getLogger(__name__)


class FasterWhisperASR:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language="ja",
            beam_size=self.cfg.beam_size,
            condition_on_previous_text=False,
            without_timestamps=True,  # we already know the utterance's timing
            vad_filter=False,  # already segmented by our own VAD
            no_speech_threshold=self.cfg.no_speech_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
        )
        parts = []
        for seg in segments:  # lazy generator: decoding happens here
            text = seg.text.strip()
            if any(phrase in text for phrase in self.cfg.blocklist):
                log.debug("blocked hallucination: %s", text)
                continue
            parts.append(text)
        return "".join(parts)
