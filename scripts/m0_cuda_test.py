"""M0: transcribe a short Japanese audio file with faster-whisper on the GPU.

Usage:
    uv run python scripts/m0_cuda_test.py test_ja.wav [--model MODEL]
"""

import argparse
import subprocess
import time

import numpy as np

from auto_translate import cuda_dlls

cuda_dlls.setup()  # must run before faster_whisper is imported

from faster_whisper import WhisperModel  # noqa: E402  (import after DLL setup on purpose)

DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"
SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 via ffmpeg (same format M1's ingest produces).

    We avoid faster-whisper's built-in decoder (PyAV), whose API changed in newer releases.
    """
    pcm = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    # s16le = signed 16-bit little-endian ints; Whisper wants floats in [-1, 1].
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to a Japanese audio file (wav/mp3/m4a/...)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()

    t0 = time.perf_counter()
    # First run downloads the model (~1.5 GB) into the Hugging Face cache.
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    audio = load_audio(args.audio)

    t0 = time.perf_counter()
    # transcribe() returns a lazy generator; the work happens while we iterate it.
    segments, info = model.transcribe(
        audio,
        language="ja",
        beam_size=5,
        condition_on_previous_text=False,
    )
    for seg in segments:
        print(f"[{seg.start:6.1f}–{seg.end:6.1f}] {seg.text}")
    elapsed = time.perf_counter() - t0

    print(f"\naudio {info.duration:.1f}s, transcribed in {elapsed:.2f}s "
          f"(real-time factor {elapsed / info.duration:.3f}; < 1 means faster than real time)")


if __name__ == "__main__":
    main()
