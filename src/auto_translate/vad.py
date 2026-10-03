"""Silero VAD segmenter: AudioChunk stream → Utterance on each pause (or at max length).

Reuses the Silero ONNX model bundled with faster-whisper (no PyTorch needed), but keeps
the model's recurrent state across calls so it works on a live stream.

Silero misses voices mixed with loud music (narrated videos, singing). So audio it calls
non-speech isn't thrown away: every `fallback_s` it goes to Whisper as a "probe" utterance,
and Whisper (with stricter filters, see asr) decides whether anyone is talking.
"""

import logging
import queue
import time
from collections import deque

import numpy as np
from faster_whisper.vad import get_vad_model

from .config import VadConfig
from .types import SAMPLE_RATE, AudioChunk, Utterance

log = logging.getLogger(__name__)

FRAME = 512  # samples per VAD decision (32 ms)
CONTEXT = 64  # samples of the previous frame the model also sees


class StreamingSileroVAD:
    def __init__(self) -> None:
        self._session = get_vad_model().session  # onnxruntime session, CPU
        self._h = np.zeros((1, 1, 128), dtype=np.float32)  # LSTM state, carried over
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """audio: float32, length a multiple of FRAME. Returns one speech probability per frame."""
        frames = audio.reshape(-1, FRAME)
        # Each model input row = last 64 samples of the previous frame + the frame itself.
        prev_tails = np.concatenate([self._context[None, :], frames[:-1, -CONTEXT:]])
        batch = np.concatenate([prev_tails, frames], axis=1)
        probs, self._h, self._c = self._session.run(
            None, {"input": batch, "h": self._h, "c": self._c}
        )
        self._context = frames[-1, -CONTEXT:].copy()
        return probs


def _ms_to_frames(ms: float) -> int:
    return max(1, round(ms * SAMPLE_RATE / 1000 / FRAME))


class Segmenter:
    """State machine over 32 ms frames: idle → speech → (pause | max length) → emit."""

    def __init__(self, cfg: VadConfig) -> None:
        self.cfg = cfg
        self.min_silence = _ms_to_frames(cfg.min_silence_ms)
        self.pad = _ms_to_frames(cfg.speech_pad_ms)
        self.min_speech = _ms_to_frames(cfg.min_speech_ms)
        self.max_frames = _ms_to_frames(cfg.max_utterance_s * 1000)
        self.split_search = _ms_to_frames(2000)  # look this far back for a quiet split point
        self.fallback_frames = _ms_to_frames(cfg.fallback_s * 1000) if cfg.fallback_s > 0 else 0
        self.fallback_min = _ms_to_frames(cfg.fallback_min_s * 1000)
        self.fallback_min_rms = 10 ** (cfg.fallback_min_db / 20)
        self._reset(0)

    def _reset(self, next_sample: int) -> None:
        """Fresh stream state, starting at sample `next_sample` (0 = take it from the first chunk)."""
        self.vad = StreamingSileroVAD()
        self._leftover = np.zeros(0, dtype=np.float32)  # samples not yet a whole frame
        self._next_frame_sample = next_sample  # sample index of the next frame to process
        self._preroll: deque[np.ndarray] = deque(maxlen=self.pad)  # recent non-speech frames
        self._idle: list[np.ndarray] = []  # non-speech frames since the last utterance (fallback)
        self._idle_start = next_sample
        self._frames: list[np.ndarray] = []  # current utterance
        self._probs: list[float] = []
        self._start_sample = 0
        self._silence = 0  # consecutive silent frames at the end of the utterance
        self._in_speech = False
        self._read_at = 0.0
        self._out: list[Utterance] = []

    def feed(self, chunk: AudioChunk) -> list[Utterance]:
        if chunk.reset:  # vod jump: forget the old position, including any half-finished utterance
            self._reset(chunk.start_sample)
            return []
        self._read_at = chunk.read_at
        if self._next_frame_sample == 0:
            self._next_frame_sample = chunk.start_sample  # VOD may start mid-video
        audio = np.concatenate([self._leftover, chunk.pcm])
        whole = len(audio) // FRAME * FRAME
        self._leftover = audio[whole:]
        if whole:
            probs = self.vad(audio[:whole])
            for i, prob in enumerate(probs):
                self._step(audio[i * FRAME:(i + 1) * FRAME], float(prob))
                self._next_frame_sample += FRAME
        out, self._out = self._out, []
        return out

    def flush(self) -> list[Utterance]:
        """End of stream: emit whatever speech is in progress."""
        if self._in_speech:
            self._emit(len(self._frames), keep_rest=False)
        self._emit_probe(len(self._idle))
        out, self._out = self._out, []
        return out

    def _step(self, frame: np.ndarray, prob: float) -> None:
        cfg = self.cfg
        if not self._in_speech:
            if prob >= cfg.threshold:
                # Idle audio before the preroll goes to Whisper; the preroll joins the utterance.
                self._emit_probe(len(self._idle) - len(self._preroll))
                self._in_speech = True
                self._frames = [*self._preroll, frame]
                self._probs = [0.0] * len(self._preroll) + [prob]
                self._start_sample = self._next_frame_sample - len(self._preroll) * FRAME
                self._silence = 0
            else:
                self._preroll.append(frame)
                self._add_idle(frame)
            return

        self._frames.append(frame)
        self._probs.append(prob)
        self._silence = self._silence + 1 if prob < cfg.neg_threshold else 0

        if self._silence >= self.min_silence:
            # Pause found: keep `pad` frames of trailing silence, the rest becomes preroll.
            n = min(len(self._frames), len(self._frames) - self._silence + self.pad)
            self._emit(n, keep_rest=False)
        elif len(self._frames) >= self.max_frames:
            # Too long: split at the quietest frame in the last ~2 s, keep the rest going.
            lo = len(self._frames) - self.split_search
            cut = lo + int(np.argmin(self._probs[lo:])) + 1
            self._emit(cut, keep_rest=True)

    def _emit(self, n: int, keep_rest: bool) -> None:
        frames, probs = self._frames[:n], self._probs[:n]
        rest_frames, rest_probs = self._frames[n:], self._probs[n:]
        start = self._start_sample
        end = start + n * FRAME

        speech_frames = sum(p >= self.cfg.threshold for p in probs)
        if speech_frames >= self.min_speech:
            utt = Utterance(start / SAMPLE_RATE, end / SAMPLE_RATE, np.concatenate(frames))
            # Estimate when the utterance's last sample arrived: the current chunk's read time,
            # minus the audio processed after it (the trailing silence we waited for).
            processed_end = self._next_frame_sample + FRAME
            utt.timings.audio_read = self._read_at - (processed_end - end) / SAMPLE_RATE
            utt.timings.segmented = time.monotonic()
            self._out.append(utt)

        if keep_rest:
            self._frames, self._probs = rest_frames, rest_probs
            self._start_sample = end
        else:
            self._in_speech = False
            self._frames, self._probs = [], []
            self._preroll.clear()
            self._preroll.extend(rest_frames)  # deque(maxlen) keeps only the last `pad`
            self._idle, self._idle_start = list(rest_frames), end

    def _add_idle(self, frame: np.ndarray) -> None:
        if not self.fallback_frames:
            return
        if not self._idle:
            self._idle_start = self._next_frame_sample
        self._idle.append(frame)
        if len(self._idle) >= self.fallback_frames:
            # Cut at the quietest frame in the last ~2 s, so a word is less likely to be split.
            lo = max(self.fallback_min, len(self._idle) - self.split_search)
            rms = [float(np.sqrt(np.mean(f * f))) for f in self._idle[lo:]]
            self._emit_probe(lo + int(np.argmin(rms)) + 1)

    def _emit_probe(self, n: int) -> None:
        """Send the first `n` idle frames to Whisper as a probe (if long and loud enough)."""
        frames, self._idle = self._idle[:max(n, 0)], self._idle[max(n, 0):]
        start = self._idle_start
        self._idle_start += len(frames) * FRAME
        if not self.fallback_frames or len(frames) < self.fallback_min:
            return
        audio = np.concatenate(frames)
        if np.sqrt(np.mean(audio * audio)) < self.fallback_min_rms:
            return  # (near) silence
        end = start + len(audio)
        utt = Utterance(start / SAMPLE_RATE, end / SAMPLE_RATE, audio, probe=True)
        utt.timings.audio_read = self._read_at - (self._next_frame_sample + FRAME - end) / SAMPLE_RATE
        utt.timings.segmented = time.monotonic()
        self._out.append(utt)


def run_segmenter(cfg: VadConfig, in_q: "queue.Queue[AudioChunk | None]",
                  out_q: "queue.Queue[Utterance | None]") -> None:
    """Thread body."""
    seg = Segmenter(cfg)
    while (chunk := in_q.get()) is not None:  # `:=` assigns and tests in one go
        for utt in seg.feed(chunk):
            out_q.put(utt)
    for utt in seg.flush():
        out_q.put(utt)
    out_q.put(None)
