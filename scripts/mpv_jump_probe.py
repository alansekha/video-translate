"""Debug helper: while `auto-translate --mpv` plays a VOD, press jump keys and watch the
audio state second by second after each jump."""

import sys
import time

from auto_translate.output.mpv import IPC_PATH, MpvIpc

ipc = MpvIpc(IPC_PATH)
for key in sys.argv[1:] or ["RIGHT", "LEFT"]:
    before = ipc.get("time-pos")
    t0 = time.monotonic()
    ipc.command("keypress", key)
    time.sleep(1.0)  # let the jump begin (mpv pauses and reloads)
    playing_at = audio_at = None
    while time.monotonic() - t0 < 15 and audio_at is None:
        if playing_at is None and not ipc.get("pause") and ipc.get("time-pos") is not None:
            playing_at = time.monotonic() - t0
        if playing_at is not None and ipc.get("audio-pts") is not None:
            audio_at = time.monotonic() - t0
        time.sleep(0.05)
    print(f"{key:<12} {before:7.1f} → {ipc.get('time-pos'):7.1f}  playing after {playing_at}s, "
          f"audio after {audio_at}s", flush=True)
    time.sleep(4)
