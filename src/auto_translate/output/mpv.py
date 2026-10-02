"""M4: play the stream in mpv a few seconds behind live, with subtitles shown exactly on time.

mpv listens on a local TCP port (paused); ingest's ffmpeg streams video+audio into it, and mpv
buffers in memory. Once `delay_s` of audio has gone into the pipeline we unpause, so playback runs
`delay_s` behind and every subtitle is ready before its line is spoken. A scheduler thread polls
mpv's time-pos over JSON IPC (~10x/s) and draws the current line with `osd-overlay`.
"""

import json
import logging
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import MpvConfig
from ..ingest import Ingest, fmt_hms
from ..types import Subtitle

log = logging.getLogger(__name__)

IPC_PATH = r"\\.\pipe\auto-translate-mpv" if sys.platform == "win32" else "/tmp/auto-translate-mpv.sock"
WINDOWS_DEFAULT = Path(r"C:\Program Files\MPV Player\mpv.exe")
OVERLAY_ID = 1
INPUT_CONF = Path(__file__).with_name("mpv_input.conf")
KEYS_HINT = ("←/→: jump 30s · shift+←/→: 5 min · space: pause · m: mute · "
             "↑/↓: volume · f: fullscreen · q: quit")
JUMP_MESSAGE = "auto-translate-jump"  # sent by the jump keys in mpv_input.conf


class MpvError(Exception):
    pass


class MpvIpc:
    """mpv's JSON IPC: one JSON object per line. Named pipe on Windows, Unix socket elsewhere."""

    def __init__(self, path: str, timeout_s: float = 15.0) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if sys.platform == "win32":
                    self._f = open(path, "r+b", buffering=0)
                else:
                    sock = socket.socket(socket.AF_UNIX)
                    sock.connect(path)
                    self._f = sock.makefile("rwb", buffering=0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        # One request at a time: on Windows a synchronous pipe can't read and write concurrently.
        self._lock = threading.Lock()
        self._next_id = 0
        self.events: queue.Queue[dict] = queue.Queue()  # events that arrived while waiting

    def command(self, *args: Any, **named: Any) -> Any:
        """command("set_property", "pause", False) or command("osd-overlay", id=1, ...) (named args)."""
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            cmd = {"name": args[0], **named} if named else list(args)
            self._f.write((json.dumps({"command": cmd, "request_id": rid}) + "\n").encode())
            while True:
                line = self._f.readline()
                if not line:
                    raise ConnectionError("mpv closed the IPC connection")
                msg = json.loads(line)
                if msg.get("request_id") == rid:
                    if msg.get("error") != "success":
                        raise MpvError(msg.get("error"))
                    return msg.get("data")
                if "event" in msg:
                    self.events.put(msg)

    def get(self, prop: str) -> Any:
        """A property's value, or None while it's unavailable (e.g. time-pos before playback)."""
        try:
            return self.command("get_property", prop)
        except MpvError:
            return None

    def close(self) -> None:
        self._f.close()


@dataclass
class _Line:
    sub: Subtitle
    show_from: float  # timeline seconds (same space as Subtitle.start_s)
    show_until: float


def find_mpv(configured: str) -> str:
    if configured:
        return configured
    if found := shutil.which("mpv"):
        return found
    if sys.platform == "win32" and WINDOWS_DEFAULT.exists():
        return str(WINDOWS_DEFAULT)
    raise SystemExit("mpv not found: install it (winget install shinchiro.mpv) or set [mpv] path")


class MpvOutput:
    def __init__(self, cfg: MpvConfig, ingest: Ingest, stop: threading.Event) -> None:
        self.cfg = cfg
        self.ingest = ingest
        self.stop = stop  # set when the user closes mpv: the whole app should stop
        self.listen_url = f"tcp://127.0.0.1:{cfg.port}"
        self._lines: list[_Line] = []
        self._lock = threading.Lock()
        self._now: float | None = None  # current playback position on our timeline
        self._shown = ""
        self._lags: list[float] = []  # per shown line: display time - spoken start (sync quality)
        self._session_start = ingest.source.start_s  # timeline position the current session began at

        self._proc = subprocess.Popen([
            find_mpv(cfg.path), f"{self.listen_url}?listen=1",
            "--pause",  # buffer first; unpaused by the scheduler
            f"--input-ipc-server={IPC_PATH}",
            "--force-window=yes", "--title=auto-translate", "--keep-open=no",
            "--idle=yes",  # stay open between streams (a jump loads a new one)
            "--rebase-start-time=no",  # time-pos = the stream's own timestamps (same clock as ingest)
            "--cache=yes", "--demuxer-readahead-secs=120", "--demuxer-max-bytes=300MiB",
            "--force-seekable=yes", "--demuxer-seekable-cache=yes",  # for the start seek (see _play_session)
            # The incoming stream can't seek, and seeking or switching tracks (the on-screen buttons)
            # loses audio or breaks playback. So: no on-screen controller, and only safe keys.
            "--osc=no", "--no-input-default-bindings", f"--input-conf={INPUT_CONF}",
            # Decode video on the GPU's video unit, not the CPU (which a partly offloaded LLM keeps busy).
            f"--hwdec={cfg.hwdec}",
            "--msg-level=all=warn",
        ])
        self.ipc = MpvIpc(IPC_PATH)
        # mpv opens its listening socket while loading the file; wait for that before ffmpeg connects.
        deadline = time.monotonic() + 10
        while self.ipc.get("path") is None and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(0.5)
        log.info("mpv ready, waiting for video on %s (delay %.1fs)", self.listen_url, cfg.delay_s)
        self._thread = threading.Thread(target=self._run, name="mpv", daemon=True)
        self._thread.start()

    # ---- Output interface ----

    def show(self, sub: Subtitle) -> None:
        cfg = self.cfg
        if sub.end_s < self._session_start:
            return  # still in the pipeline from before a jump: belongs to the old position
        until = max(sub.end_s, sub.start_s + cfg.min_display_s) + cfg.hold_s
        line = _Line(sub, sub.start_s, until)
        now = self._now
        if now is not None and now > until:
            # Arrived after its moment (delay too short for this line): show it right away.
            log.debug("late subtitle (%.1fs): %s", now - sub.end_s, sub.ja)
            line = _Line(sub, now, now + cfg.min_display_s + cfg.hold_s)
        with self._lock:
            self._lines.append(line)

    def drain(self) -> None:
        """Stream ended normally: let mpv play out its buffered `delay_s` before closing."""
        deadline = time.monotonic() + self.cfg.delay_s + 15
        while not self.stop.is_set() and time.monotonic() < deadline:
            if self._proc.poll() is not None or self.ipc.get("idle-active"):
                break
            time.sleep(0.5)

    def close(self) -> None:
        self.stop.set()
        if lags := self._lags:
            on_time = sum(abs(x) <= 0.5 for x in lags)
            log.info("mpv sync: %d lines shown, %d within 0.5s of speech, avg %+.2fs, worst %+.2fs",
                     len(lags), on_time, sum(lags) / len(lags), max(lags, key=abs))
        try:
            self.ipc.command("quit")
        except (ConnectionError, OSError, MpvError):
            pass
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    # ---- scheduler ----

    def _run(self) -> None:
        try:
            first = True
            while not self.stop.is_set():
                self._play_session(show_hint=first)  # returns after a jump (or on stop)
                first = False
        except (ConnectionError, OSError, json.JSONDecodeError):
            if not self.stop.is_set():
                log.info("mpv was closed: stopping")
                self.stop.set()

    def _play_session(self, show_hint: bool) -> None:
        """One ffmpeg session: buffer, unpause, draw subtitles until the user jumps or we stop."""
        # Wait for the first audio timestamp: it maps our timeline onto mpv's time-pos.
        while not self.ingest.first_pts_ready.wait(0.2):
            if self.stop.is_set():
                return
        assert self.ingest.first_pts is not None
        offset = self.ingest.first_pts - self.ingest.source.start_s  # pts = timeline + offset
        self._session_start = self.ingest.source.start_s
        # VOD: ingest bursts delay_s of audio at once, so a short wait is enough for the first
        # lines to be ready. Live: audio only arrives at 1x, so wait out the whole delay.
        wait = self.cfg.delay_s if self.ingest.source.live else self.cfg.start_wait_s
        unpause_at = self.ingest.first_pts_at + wait
        paused = True
        while not self.stop.is_set():
            if paused and time.monotonic() >= unpause_at:
                # Video starts at the keyframe before the start point (up to ~5 s earlier) while
                # audio starts exactly on it; mpv would play that video silently. Skip it with an
                # exact seek to the first audio sample (inside mpv's buffer, so no stream seek).
                self.ipc.command("seek", self.ingest.first_pts, "absolute+exact")
                self.ipc.command("set_property", "pause", False)
                if show_hint:
                    self.ipc.command("show-text", KEYS_HINT, 6000)
                paused = False
                log.info("mpv playing from %s", fmt_hms(self._session_start))
            if (pos := self.ipc.get("time-pos")) is not None:
                self._now = pos - offset
                self._render(self._now)
            if (delta := self._jump_request()) and self._jump(delta):
                return
            time.sleep(0.1)

    def _jump_request(self) -> float:
        """Sum of jump keys pressed since the last check (mpv sends them as client-messages)."""
        delta = 0.0
        while not self.ipc.events.empty():
            ev = self.ipc.events.get()
            args = ev.get("args") or []
            if ev.get("event") == "client-message" and args[:1] == [JUMP_MESSAGE]:
                delta += float(args[1])
        return delta

    def _jump(self, delta: float) -> bool:
        """Restart ingest + mpv at (current position + delta). Returns False if not possible."""
        src = self.ingest.source
        if src.live:
            self.ipc.command("show-text", "Jumping only works for VODs", 2000)
            return False
        if self._now is None:  # not playing yet
            return False
        target = max(0.0, self._now + delta)
        if src.duration:
            target = min(target, max(0.0, src.duration - 5))
        log.info("jump %+.0fs → %s", delta, fmt_hms(target))
        self.ipc.command("set_property", "pause", True)
        self._now = None
        self._shown = ""
        self.ipc.command("osd-overlay", id=OVERLAY_ID, format="none", data="")
        with self._lock:
            self._lines.clear()
        self.ingest.request_restart(target)  # stops the old ffmpeg
        self.ipc.command("loadfile", f"{self.listen_url}?listen=1")  # listen for the new one
        self._wait_for_event("start-file")
        time.sleep(0.5)
        self.ipc.command("show-text", f"Jumping to {fmt_hms(target)}…", 4000)
        self.ingest.go()
        return True

    def _wait_for_event(self, name: str, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.ipc.get("pause")  # any request pumps pending events into ipc.events
            while not self.ipc.events.empty():
                if self.ipc.events.get().get("event") == name:
                    return
            time.sleep(0.05)

    def _render(self, now: float) -> None:
        with self._lock:
            self._lines = [ln for ln in self._lines if ln.show_until > now - 60]  # forget old lines
            active = [ln for ln in self._lines if ln.show_from <= now < ln.show_until]
        # Several can overlap (hold time); the most recent one wins, like normal subtitles.
        current = max(active, key=lambda ln: ln.show_from) if active else None
        text = _ass(current.sub, self.cfg) if current else ""
        if text != self._shown:
            self._shown = text
            if current:
                self._lags.append(now - current.sub.start_s)  # >0: shown after the line started
            self.ipc.command("osd-overlay", id=OVERLAY_ID, format="ass-events" if text else "none",
                             data=text, res_x=0, res_y=720)


def _ass(sub: Subtitle, cfg: MpvConfig) -> str:
    """Subtitle → ASS markup: bottom-centred, JA small grey above EN large white, black outline."""

    def esc(s: str) -> str:  # braces and backslashes are ASS syntax; swap for full-width lookalikes
        return s.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ")

    head = r"{\an2\bord2.5\shad0\3c&H000000&}"
    ja = rf"{{\fs{cfg.ja_size}\1c&HC8C8C8&}}{esc(sub.ja)}"
    if not sub.en:
        return head + ja
    en = rf"{{\fs{cfg.en_size}\1c&HFFFFFF&}}{esc(sub.en)}"
    return head + ja + r"\N" + en
