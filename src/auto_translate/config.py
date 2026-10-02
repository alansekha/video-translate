"""Load config.toml into typed dataclasses (missing keys fall back to the defaults below)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IngestConfig:
    # Live streams usually have no audio-only format, so fall back to 93 (360p, 128 kbps audio).
    ytdlp_format: str = "bestaudio/93/best"
    cookies_from_browser: str = ""  # e.g. "chrome" for members-only streams


@dataclass
class VadConfig:
    threshold: float = 0.5  # speech probability that starts an utterance
    neg_threshold: float = 0.35  # below this counts as silence
    min_silence_ms: int = 500  # pause length that ends an utterance
    speech_pad_ms: int = 200  # audio kept before/after speech
    min_speech_ms: int = 250  # drop utterances with less speech than this
    max_utterance_s: float = 9.0  # force a split (at the quietest point) after this


@dataclass
class AsrConfig:
    backend: str = "faster-whisper"
    model: str = "kotoba-tech/kotoba-whisper-v2.0-faster"
    device: str = "cuda"
    compute_type: str = "int8_float16"  # float16 needs ~1 GB more VRAM for the same accuracy
    beam_size: int = 5
    # A segment is skipped when no_speech_prob > no_speech_threshold AND avg_logprob < log_prob_threshold.
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    blocklist: list[str] = field(default_factory=list)  # known hallucinations, dropped if contained


@dataclass
class Ct2NmtConfig:
    """A CTranslate2 seq2seq model (Opus-MT, Sugoi) + SentencePiece tokenizers."""

    model: str  # HF repo id or local folder
    source_spm: str  # tokenizer paths, relative to the model folder
    target_spm: str
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 4


@dataclass
class OpenAICompatConfig:
    """Any server with an OpenAI-style /v1/chat/completions endpoint (Ollama, llama-server)."""

    model: str
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds ~2 s per request.
    base_url: str = "http://127.0.0.1:11434/v1"
    temperature: float = 0.2
    max_tokens: int = 120
    timeout_s: float = 20.0
    load_timeout_s: float = 300.0  # first request loads the model into VRAM (can take > 1 min)
    extra_body: dict = field(default_factory=dict)  # merged into the request JSON as-is


@dataclass
class TranslateConfig:
    preset: str = "balanced"  # key in `presets`, or "none" to only transcribe
    context_lines: int = 4  # previous JA/EN lines given to the LLM
    glossary: str = "glossary.toml"
    # name → {"backend": "ct2" | "openai-compat", ...settings for that backend}
    presets: dict[str, dict] = field(default_factory=dict)


@dataclass
class OutputConfig:
    terminal: bool = True
    web: bool = True
    web_host: str = "0.0.0.0"  # all interfaces: phones / the Mac on the LAN can connect
    web_port: int = 8765


@dataclass
class MpvConfig:
    """M4: play the stream in mpv `delay_s` behind live, with subtitles drawn in sync."""

    enabled: bool = False  # or pass --mpv
    path: str = ""  # mpv executable; "" = PATH, then C:\Program Files\MPV Player\mpv.exe
    # Head start the pipeline gets over playback. Must cover the longest utterance
    # (vad.max_utterance_s) + processing (~1-3 s).
    delay_s: float = 9.0
    # VOD: delay_s of audio is read at full speed at (re)start, so playback can begin after
    # this short wait instead of delay_s (also makes jumps quick). Live always waits delay_s.
    start_wait_s: float = 3.0
    port: int = 9137  # local TCP port ffmpeg streams the video to
    # mpv video decoding: "auto-safe" = hardware (NVDEC / D3D11VA) when available, "no" = CPU.
    hwdec: str = "auto-safe"
    # HLS (m3u8) H.264 + AAC, at most 480p: fast to fetch, cheap to decode, fits in MPEG-TS.
    ytdlp_format: str = ("bv*[height<=480][vcodec^=avc1][protocol^=m3u8]+ba[protocol^=m3u8]"
                         "/b[height<=480][protocol^=m3u8]/b[protocol^=m3u8]")
    en_size: int = 44  # font sizes on a 720-line canvas
    ja_size: int = 28
    min_display_s: float = 1.5  # show short lines at least this long
    hold_s: float = 1.0  # keep a line up a bit after it ends (unless the next one starts)


@dataclass
class Config:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mpv: MpvConfig = field(default_factory=MpvConfig)
    stats_interval_s: float = 30.0


def load_config(path: Path) -> Config:
    if not path.exists():
        return Config()
    with path.open("rb") as f:
        data = tomllib.load(f)
    # `**dict` passes the TOML table as keyword arguments; unknown keys raise a TypeError.
    return Config(
        ingest=IngestConfig(**data.get("ingest", {})),
        vad=VadConfig(**data.get("vad", {})),
        asr=AsrConfig(**data.get("asr", {})),
        translate=TranslateConfig(**data.get("translate", {})),
        output=OutputConfig(**data.get("output", {})),
        mpv=MpvConfig(**data.get("mpv", {})),
        stats_interval_s=data.get("stats_interval_s", 30.0),
    )
