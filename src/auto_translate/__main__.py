"""CLI entry point: `uv run auto-translate <youtube-url | audio file> [options]`."""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

from . import cuda_dlls


def main() -> None:
    parser = argparse.ArgumentParser(prog="auto-translate")
    parser.add_argument("source", help="YouTube URL (live or VOD) or a local audio/video file")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--mode", choices=["auto", "live", "vod"], default="auto",
                        help="auto (default) asks yt-dlp whether the URL is live right now")
    parser.add_argument("--start", metavar="TIME",
                        help="vod: start here (962, 16:02, 1:02:03, 16m2s); default: the URL's t= value")
    parser.add_argument("--fast", action="store_true",
                        help="vod: process as fast as possible instead of at 1x speed")
    parser.add_argument("--preset", help="translation preset from config.toml (e.g. light, balanced); "
                                         "'none' = transcribe only")
    parser.add_argument("--no-web", action="store_true", help="don't start the browser viewer")
    parser.add_argument("--mpv", action="store_true",
                        help="watch in mpv, delayed so subtitles appear exactly when each line is spoken")
    parser.add_argument("--delay", type=float, metavar="SECONDS",
                        help="mpv: how far playback stays behind the pipeline "
                             "(default: the preset's mpv_delay_s, else [mpv] delay_s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):  # Japanese text / '·' on Windows, even when redirected
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("faster_whisper", "httpx"):  # they log every utterance / HTTP request
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cuda_dlls.setup()  # before anything imports faster_whisper / ctranslate2

    # Imported here (not at the top) so the DLL setup above runs first.
    from .asr import create_asr
    from .config import load_config
    from .glossary import Glossary
    from .ingest import Ingest, resolve_source
    from .output.base import Output
    from .output.terminal import TerminalOutput
    from .output.web import WebOutput
    from .pipeline import run
    from .translate import create_translator

    cfg = load_config(args.config)
    if args.preset:
        cfg.translate.preset = args.preset
    # Slow presets need more head start in mpv: --delay > preset's mpv_delay_s > [mpv] delay_s.
    preset_delay = cfg.translate.presets.get(cfg.translate.preset, {}).get("mpv_delay_s")
    if args.delay is not None:
        cfg.mpv.delay_s = args.delay
    elif preset_delay is not None:
        cfg.mpv.delay_s = float(preset_delay)
    log = logging.getLogger("auto_translate")
    source = resolve_source(args.source, args.mode, args.start, args.fast, cfg.ingest)

    # Load the models before starting ingest, so no audio backlog builds up during loading.
    t0 = time.perf_counter()
    asr = create_asr(cfg.asr)
    log.info("ASR ready (%s, %s) in %.1fs", cfg.asr.model, cfg.asr.device, time.perf_counter() - t0)

    t0 = time.perf_counter()
    glossary = Glossary.load(Path(cfg.translate.glossary))
    translator = create_translator(cfg.translate, glossary)
    log.info("translator ready (preset %s, %d glossary terms) in %.1fs",
             cfg.translate.preset, len(glossary.terms), time.perf_counter() - t0)

    use_mpv = args.mpv or cfg.mpv.enabled
    if use_mpv and not source.paced:
        raise SystemExit("--mpv plays at 1x speed; it can't be combined with --fast")
    video_out = f"tcp://127.0.0.1:{cfg.mpv.port}" if use_mpv else None
    burst_s = cfg.mpv.delay_s if use_mpv and not source.live else 0.0
    ingest = Ingest(source, cfg.ingest, video_out, cfg.mpv.ytdlp_format, burst_s)
    stop = threading.Event()

    outputs: list[Output] = []
    if cfg.output.terminal:
        outputs.append(TerminalOutput())
    if cfg.output.web and not args.no_web:
        outputs.append(WebOutput(cfg.output.web_host, cfg.output.web_port))
    if use_mpv:
        from .output.mpv import MpvOutput

        outputs.append(MpvOutput(cfg.mpv, ingest, stop))  # starts mpv; must listen before ingest

    run(ingest, cfg, asr, translator, outputs, stop)


if __name__ == "__main__":
    main()
