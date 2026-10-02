"""yt-dlp | ffmpeg → 16 kHz mono PCM chunks on a queue.

Two modes:
- live: audio arrives at 1x on its own; timestamps = seconds since ingest start.
- vod:  (archived stream, video, or local file) we may start mid-video and pace reading at 1x;
        timestamps = position in the video, so they match the YouTube player.
"""

import logging
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO
from urllib.parse import parse_qs, urlparse

import numpy as np

from .config import IngestConfig
from .types import SAMPLE_RATE, AudioChunk

log = logging.getLogger(__name__)

CHUNK_SAMPLES = 4096  # 256 ms; a multiple of the VAD frame size (512)
_PTS = re.compile(r"pts_time:(-?[\d.]+)")


@dataclass
class Source:
    location: str  # URL or local path
    live: bool
    start_s: float = 0.0  # vod: where to start in the video
    paced: bool = True  # vod: read at 1x speed (False = as fast as possible)
    duration: float | None = None  # vod: length in seconds, if known


# ---- source resolution -------------------------------------------------------------------

_HMS = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s?)?$")


def parse_time(text: str) -> float:
    """'962', '962s', '16:02', '1:02:03', '16m2s', '1h2m3s' → seconds."""
    text = text.strip().lower()
    if ":" in text:
        seconds = 0.0
        for part in text.split(":"):  # 1:02:03 → ((1*60)+2)*60+3
            seconds = seconds * 60 + float(part)
        return seconds
    m = _HMS.match(text)
    if not text or not m:
        raise ValueError(f"bad time {text!r}; use e.g. 962, 16:02, 1:02:03 or 16m2s")
    h, mins, s = m.groups()
    return int(h or 0) * 3600 + int(mins or 0) * 60 + float(s or 0)


def start_from_url(url: str) -> float:
    """Read YouTube's `t=` (or `start=`) parameter, e.g. ...watch?v=ID&t=962s → 962.0."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query) | parse_qs(parsed.fragment)  # `|` merges two dicts
    for key in ("t", "start"):
        if key in params:
            return parse_time(params[key][0])
    return 0.0


def probe(url: str, cfg: IngestConfig) -> tuple[str, float | None]:
    """Ask yt-dlp for (live_status, duration in seconds or None).

    live_status: 'is_live', 'was_live', 'not_live', 'is_upcoming' or 'post_live'.
    """
    cmd = ["yt-dlp", "--no-warnings", "--skip-download", "--print", "%(live_status)s|%(duration)s"]
    cmd += _cookie_args(cfg) + [url]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    status, _, duration = out.partition("|")  # split at the first '|'
    return status, float(duration) if duration not in ("", "NA", "None") else None


def fmt_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def resolve_source(location: str, mode: str, start: str | None, fast: bool,
                   cfg: IngestConfig) -> Source:
    """Turn CLI arguments into a Source. mode: 'auto' | 'live' | 'vod'."""
    is_file = Path(location).exists()
    duration: float | None = None
    if mode == "auto":
        if is_file:
            mode = "vod"
        else:
            status, duration = probe(location, cfg)
            log.info("yt-dlp: live_status=%s duration=%s", status,
                     fmt_hms(duration) if duration else "?")
            if status == "is_upcoming":
                raise SystemExit("this stream hasn't started yet")
            mode = "live" if status == "is_live" else "vod"

    if mode == "live":
        if start or start_from_url(location):
            log.warning("live mode: ignoring the start time (we always start at the live edge)")
        return Source(location, live=True)

    start_s = parse_time(start) if start else (0.0 if is_file else start_from_url(location))
    if duration and start_s >= duration:
        raise SystemExit(f"start {fmt_hms(start_s)} is past the end of the video ({fmt_hms(duration)}); "
                         "note h:mm:ss vs mm:ss, e.g. 6:00 = 6 minutes, 6:00:00 = 6 hours")
    return Source(location, live=False, start_s=start_s, paced=not fast, duration=duration)


# ---- ingest ------------------------------------------------------------------------------


def _cookie_args(cfg: IngestConfig) -> list[str]:
    return ["--cookies-from-browser", cfg.cookies_from_browser] if cfg.cookies_from_browser else []


class Ingest:
    def __init__(self, source: Source, cfg: IngestConfig,
                 video_out: str | None = None, video_format: str = "", burst_s: float = 0.0):
        """video_out: if set (e.g. tcp://127.0.0.1:9137), also send video+audio there for mpv (M4).
        burst_s: (video mode, vod) read this much at full speed before pacing at 1x, so the
        pipeline gets its head start in ~1 s instead of waiting it out in real time."""
        self.source = source
        self.cfg = cfg
        self.video_out = video_out
        self.video_format = video_format
        self.burst_s = burst_s
        self._procs: list[subprocess.Popen] = []
        self._inputs: list[str] | None = None  # media URLs from yt-dlp, reused across jumps
        self._stopped = False
        self._session = 0  # counts ffmpeg runs (1 + number of jumps)
        # Presentation timestamp (pts) of the first PCM sample: links our timeline to mpv's time-pos.
        self.first_pts: float | None = None
        self.first_pts_at = 0.0  # time.monotonic() when it was seen
        self.first_pts_ready = threading.Event()
        # Jumping (vod): restart at a new position without ending the pipeline.
        self._restart_at: float | None = None
        self._go = threading.Event()  # set when the new session may start (mpv is listening again)

    def request_restart(self, start_s: float) -> None:
        """Begin a jump: stop the current ffmpeg; the new one starts after go()."""
        self._go.clear()
        self.first_pts, self._restart_at = None, start_s
        self.first_pts_ready.clear()
        self._terminate()

    def go(self) -> None:
        self._go.set()

    def _commands(self) -> list[list[str]]:
        if self.video_out:
            return [self._video_command()]
        return self._audio_commands()

    def _video_command(self) -> list[str]:
        """One ffmpeg, two outputs: video+audio (stream copy) to mpv, and PCM for the pipeline.

        ffmpeg reads YouTube's HLS URLs directly (fetched in small segments, so not throttled).
        -copyts keeps the original timestamps in both outputs, so both share one clock.
        """
        src = self.source
        if self._inputs is None:
            if Path(src.location).exists():
                self._inputs = [src.location]
            else:  # URLs stay valid for hours, so jumps can reuse them (saves ~4 s per jump)
                cmd = ["yt-dlp", "--no-warnings", "-f", self.video_format, "-g"]
                cmd += _cookie_args(self.cfg) + [src.location]
                self._inputs = subprocess.run(cmd, capture_output=True, text=True,
                                              check=True).stdout.split()
        inputs = self._inputs

        ffmpeg = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-loglevel", "info", "-copyts"]
        for url in inputs:  # -readrate / -ss are per-input options
            if not src.live:
                ffmpeg += ["-readrate", "1"]  # 1x, like -re
                if self.burst_s:
                    ffmpeg += ["-readrate_initial_burst", f"{self.burst_s:g}"]
                if src.start_s:
                    ffmpeg += ["-ss", f"{src.start_s:.3f}"]
            ffmpeg += ["-i", url]
        audio = f"{len(inputs) - 1}:a:0"  # separate video+audio URLs: audio is the 2nd input
        assert self.video_out is not None
        ffmpeg += ["-map", "0:v:0?", "-map", audio, "-c", "copy", "-f", "mpegts",
                   "-muxdelay", "0", "-muxpreload", "0", self.video_out]
        # ashowinfo logs each audio frame's pts; we only need the first one.
        ffmpeg += ["-map", audio, "-af", "ashowinfo",
                   "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"]
        return ffmpeg

    def _read_stderr(self, stream: IO[bytes], session: int) -> None:
        """Thread body (video mode): grab the first pts, pass real problems on to the log."""
        receiver_gone = False  # mpv closed: everything ffmpeg says after that is expected fallout
        for raw in stream:
            line = raw.decode("utf-8", errors="replace").rstrip()
            current = session == self._session  # an old ffmpeg (before a jump) must not set first_pts
            if current and self.first_pts is None and (m := _PTS.search(line)):
                self.first_pts = float(m.group(1))
                self.first_pts_at = time.monotonic()
                self.first_pts_ready.set()
                log.info("first audio pts %.3f", self.first_pts)
            elif "ashowinfo" not in line and any(w in line for w in ("rror", "arning", "failed")):
                # Windows socket errors -10053/-10054 (or EPIPE): the video receiver went away.
                receiver_gone |= any(c in line for c in ("-10053", "-10054", "Broken pipe", "Connection reset"))
                receiver_gone |= self.video_out is not None and "muxing a packet" in line
                log.log(logging.DEBUG if receiver_gone else logging.WARNING, "ffmpeg: %s", line)

    def _audio_commands(self) -> list[list[str]]:
        src = self.source
        ffmpeg = ["ffmpeg", "-nostdin", "-loglevel", "error"]
        if not src.live:
            if src.paced:
                ffmpeg.append("-re")  # read at 1x; audio skipped by -ss is not paced
            if src.start_s:
                # Input seek. On yt-dlp's pipe ffmpeg decodes and discards up to start_s,
                # which is fast because yt-dlp downloads much faster than 1x.
                ffmpeg += ["-ss", f"{src.start_s:.3f}"]
        out = ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"]

        if Path(src.location).exists():
            return [ffmpeg + ["-i", src.location] + out]

        ytdlp = ["yt-dlp", "-f", self.cfg.ytdlp_format, "--quiet", "--no-warnings", "-o", "-"]
        if not src.live:
            # yt-dlp's chunked range requests avoid YouTube's ~1.5x throttling of plain HTTP reads.
            ytdlp += ["--http-chunk-size", "10M"]
        ytdlp += _cookie_args(self.cfg) + [src.location]
        return [ytdlp, ffmpeg + ["-i", "pipe:0"] + out]

    def run(self, out_q: "queue.Queue[AudioChunk | None]") -> None:
        """Thread body: read PCM until EOF, then put None. Never drops audio.

        After a jump (request_restart + go) it starts over at the new position and sends a
        `reset` chunk first, so the segmenter drops the half-finished utterance from before.
        """
        try:
            while True:
                self._read_session(out_q)
                if self._stopped or self._restart_at is None:
                    break
                self.source = replace(self.source, start_s=self._restart_at)
                self._restart_at = None
                out_q.put(AudioChunk(round(self.source.start_s * SAMPLE_RATE),
                                     np.zeros(0, dtype=np.float32), time.monotonic(), reset=True))
                while not self._go.wait(0.2):  # until mpv listens again
                    if self._stopped:
                        return
        finally:
            out_q.put(None)
            self.stop()

    def _read_session(self, out_q: "queue.Queue[AudioChunk | None]") -> None:
        cmds = self._commands()
        procs: list[subprocess.Popen] = []
        stdin = None
        for cmd in cmds:
            stderr = subprocess.PIPE if self.video_out else None
            p = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=stderr)
            if stdin is not None:
                stdin.close()  # the child owns it now (lets yt-dlp see a broken pipe if ffmpeg exits)
            stdin = p.stdout
            procs.append(p)
        self._procs = procs
        self._session += 1
        if self.video_out:
            threading.Thread(target=self._read_stderr, args=(procs[-1].stderr, self._session),
                             name="ffmpeg-log", daemon=True).start()
        log.info("ingest started (%s): %s", self._describe(), " | ".join(c[0] for c in cmds))

        stdout = procs[-1].stdout
        assert stdout is not None
        # VOD timestamps continue from the start offset, so they match the video's clock.
        first = round(self.source.start_s * SAMPLE_RATE)
        sample = first
        while True:
            # BufferedReader.read(n) blocks until n bytes arrive or EOF.
            data = stdout.read(CHUNK_SAMPLES * 2)  # 2 bytes per s16 sample
            if not data:
                break
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            out_q.put(AudioChunk(sample, pcm, time.monotonic()))
            sample += len(pcm)
        for p in procs:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
        log.info("ingest session ended after %.1fs of audio (exit codes %s)",
                 (sample - first) / SAMPLE_RATE, [p.poll() for p in procs])
        if sample == first and self.source.start_s and self._restart_at is None:
            log.error("no audio after %s: is the start time past the end of the video?",
                      fmt_hms(self.source.start_s))

    def _describe(self) -> str:
        src = self.source
        video = f", video → {self.video_out}" if self.video_out else ""
        if src.live:
            return "live" + video
        return f"vod from {src.start_s:.1f}s, {'1x' if src.paced else 'fast'}{video}"

    def stop(self) -> None:
        self._stopped = True
        self._go.set()  # wake run() if it's waiting to restart
        self._terminate()

    def _terminate(self) -> None:
        for p in self._procs:
            if p.poll() is None:
                p.terminate()
