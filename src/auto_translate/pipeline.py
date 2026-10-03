"""Wire the stages together with threads and queues (like goroutines + channels).

    Ingest ─audio_q─▶ Segmenter ─utt_q─▶ ASR ─text_q─▶ Translate ─sub_q─▶ main thread (output)

Each queue ends with a `None` sentinel, which plays the role of closing a Go channel.
"""

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .asr.base import ASR
from .config import Config
from .ingest import Ingest
from .output.base import Output
from .translate.base import Translator
from .types import AudioChunk, Subtitle, Transcript, Utterance
from .vad import run_segmenter

log = logging.getLogger(__name__)


def run_asr(asr: ASR, in_q: "queue.Queue[Utterance | None]",
            out_q: "queue.Queue[Transcript | None]") -> None:
    """Thread body."""
    while (utt := in_q.get()) is not None:
        utt.timings.asr_start = time.monotonic()
        text = asr.transcribe(utt.audio, utt.probe)
        utt.timings.asr_end = time.monotonic()
        if text:
            out_q.put(Transcript(utt.start_s, utt.end_s, text, utt.timings))
    out_q.put(None)


def run_translate(translator: Translator | None, context_lines: int,
                  in_q: "queue.Queue[Transcript | None]",
                  out_q: "queue.Queue[Subtitle | None]") -> None:
    """Thread body. Keeps the last few (ja, en) pairs as context for the next line."""
    history: deque[tuple[str, str]] = deque(maxlen=context_lines)
    while (tr := in_q.get()) is not None:
        tr.timings.mt_start = time.monotonic()
        en = ""
        if translator is not None:
            try:
                en = translator.translate(tr.ja, list(history))
            except Exception as e:  # e.g. Ollama not running: keep the JA line, don't crash
                log.error("translation failed: %s", e)
                en = "[translation failed]"
            else:
                history.append((tr.ja, en))
        tr.timings.mt_end = time.monotonic()
        out_q.put(Subtitle(tr.start_s, tr.end_s, tr.ja, en, tr.timings))
    out_q.put(None)


@dataclass
class Stats:
    lines: int = 0
    asr_s: list[float] = field(default_factory=list)
    mt_s: list[float] = field(default_factory=list)
    total_s: list[float] = field(default_factory=list)  # end of speech → subtitle ready
    max_q: dict[str, int] = field(default_factory=dict)

    def add(self, sub: Subtitle) -> None:
        tm = sub.timings
        self.lines += 1
        self.asr_s.append(tm.asr_end - tm.asr_start)
        self.mt_s.append(tm.mt_end - tm.mt_start)
        self.total_s.append(tm.mt_end - tm.audio_read)

    def summary(self) -> str:
        if not self.lines:
            return "no lines yet"

        def fmt(name: str, xs: list[float]) -> str:
            return f"{name} avg {sum(xs) / len(xs):.2f}s max {max(xs):.2f}s"

        maxq = " ".join(f"{k}={v}" for k, v in self.max_q.items())
        return (f"{self.lines} lines · {fmt('asr', self.asr_s)} · {fmt('mt', self.mt_s)} · "
                f"{fmt('latency', self.total_s)} · max queue {maxq}")


def run(ingest: Ingest, cfg: Config, asr: ASR, translator: Translator | None,
        outputs: list[Output], stop: threading.Event) -> None:
    """Run until the stream ends, Ctrl+C, or `stop` is set (e.g. the mpv window was closed)."""
    audio_q: queue.Queue[AudioChunk | None] = queue.Queue()  # unbounded: never drop audio
    utt_q: queue.Queue[Utterance | None] = queue.Queue()
    text_q: queue.Queue[Transcript | None] = queue.Queue()
    sub_q: queue.Queue[Subtitle | None] = queue.Queue()
    queues = {"audio": audio_q, "utt": utt_q, "text": text_q}

    # daemon=True: these threads won't keep the process alive after main exits (e.g. on Ctrl+C).
    threads = [
        threading.Thread(target=ingest.run, args=(audio_q,), name="ingest", daemon=True),
        threading.Thread(target=run_segmenter, args=(cfg.vad, audio_q, utt_q), name="vad", daemon=True),
        threading.Thread(target=run_asr, args=(asr, utt_q, text_q), name="asr", daemon=True),
        threading.Thread(target=run_translate, args=(translator, cfg.translate.context_lines, text_q, sub_q),
                         name="translate", daemon=True),
    ]
    for t in threads:
        t.start()

    stats = Stats()
    started = time.monotonic()
    next_stats = started + cfg.stats_interval_s
    try:
        while not stop.is_set():
            try:
                item = sub_q.get(timeout=0.5)  # timeout keeps Ctrl+C responsive
                if item is None:
                    for out in outputs:
                        out.drain()
                    break
                stats.add(item)
                for out in outputs:
                    out.show(item)
            except queue.Empty:
                pass

            for name, q in queues.items():
                stats.max_q[name] = max(stats.max_q.get(name, 0), q.qsize())
            if (now := time.monotonic()) >= next_stats:
                next_stats = now + cfg.stats_interval_s
                depths = " ".join(f"{name}={q.qsize()}" for name, q in queues.items())
                log.info("[%.0fs] queues %s | %s", now - started, depths, stats.summary())
                for name in ("utt", "text"):
                    if queues[name].qsize() > 3:
                        log.warning("falling behind: %s queue = %d", name, queues[name].qsize())
    except KeyboardInterrupt:
        log.info("stopping (Ctrl+C)")
    finally:
        ingest.stop()
        for out in outputs:
            out.close()
        log.info("done: %s", stats.summary())
