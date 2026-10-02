# auto-translate

Live **Japanese → English subtitles** for YouTube streams (VTubers, gamers), running fully locally
on your own GPU. Japanese speech is transcribed with Whisper (kotoba-whisper) and translated by a local
NMT model or a local LLM (Ollama). Subtitles can be shown in three places, at the same time:

- **Terminal**: Japanese + English lines with timings and latency.
- **Browser viewer**: big captions on any device on your LAN (phone, Mac, tablet).
- **mpv**: the video plays a few seconds behind, so each subtitle appears exactly when the line is spoken.

```
yt-dlp → ffmpeg → VAD (Silero) → ASR (faster-whisper) → glossary + translator → terminal / browser / mpv
```

Works with **live streams** and **VODs** (past streams, normal videos), and with **local audio/video files**.

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line options](#command-line-options)
- [Live vs VOD](#live-vs-vod)
- [Translation presets](#translation-presets)
- [Outputs](#outputs)
- [mpv: synced video + subtitles](#mpv-synced-video--subtitles)
- [Glossary](#glossary)
- [Configuration reference (`config.toml`)](#configuration-reference-configtoml)
- [Reading the logs](#reading-the-logs)
- [Helper scripts](#helper-scripts)
- [Troubleshooting](#troubleshooting)

---

## Requirements

| What | Notes |
|---|---|
| Windows + NVIDIA GPU | Tested on an RTX 3060 Laptop (6 GB VRAM). Keep the laptop **plugged in** (GPU clocks drop on battery). |
| [uv](https://docs.astral.sh/uv/) | Python package/project manager. Installs Python 3.12 and all dependencies. |
| [ffmpeg](https://ffmpeg.org/) | On `PATH`. Decodes audio, and streams the video to mpv. |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | On `PATH`. **Keep it updated** (`yt-dlp -U` or `winget upgrade yt-dlp`): YouTube changes break old versions. |
| [Ollama](https://ollama.com/) | Only for the LLM presets (`balanced`, `quality`). |
| [mpv](https://mpv.io/) | Only for `--mpv`. Found on `PATH` or at `C:\Program Files\MPV Player\mpv.exe`. |

CUDA libraries (cuBLAS, cuDNN 9) come as pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`), so you
don't need to install the CUDA Toolkit. Only an up-to-date NVIDIA driver.

## Installation

```powershell
# Optional: keep the uv cache and Hugging Face models off C:
$env:UV_CACHE_DIR = "X:\uv-cache"
$env:HF_HOME      = "X:\hf-home"

uv sync                                       # creates .venv and installs dependencies

# LLM translator for the default preset
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

The Whisper model (`kotoba-tech/kotoba-whisper-v2.0-faster`, ~1.5 GB) and the NMT models are downloaded
from Hugging Face automatically on first use.

Check that the GPU works:

```powershell
uv run python scripts/m0_cuda_test.py some_japanese_clip.wav
```

## Quick start

```powershell
# Live stream: subtitles in the terminal + browser viewer
uv run auto-translate "https://www.youtube.com/watch?v=VIDEO_ID"

# Same, but watch the video in mpv with subtitles synced to speech
uv run auto-translate "https://www.youtube.com/watch?v=VIDEO_ID" --mpv

# A VOD starting at 16:02
uv run auto-translate "https://www.youtube.com/watch?v=VIDEO_ID" --start 16:02 --mpv

# A local file, as fast as possible, Japanese only (no translation)
uv run auto-translate recording.mp4 --fast --preset none
```

Stop with **Ctrl+C** (or by closing the mpv window). The first run takes longer: models are downloaded,
and Ollama loads the LLM into VRAM (~1–2 min on the first start; that's the "warm-up").

Then open the browser viewer: **http://127.0.0.1:8765/** on this PC, or `http://<this-PC's-LAN-IP>:8765/`
on a phone (the address is printed at startup).

## Command-line options

```
uv run auto-translate SOURCE [--mode auto|live|vod] [--start TIME] [--fast]
                             [--preset NAME] [--mpv] [--delay SECONDS] [--no-web] [--config FILE] [-v]
```

| Option | Default | Description |
|---|---|---|
| `SOURCE` | (required) | A YouTube URL (live or VOD, `youtube.com/watch?v=…`, `youtu.be/…`, `…/live/…`), or a path to a local audio/video file. Quote URLs in PowerShell (`&` is special). |
| `--mode auto\|live\|vod` | `auto` | `auto` asks yt-dlp whether the URL is live right now (~4 s). `live` / `vod` skip that check. Local files are always `vod`. |
| `--start TIME` | URL's `t=` or 0 | VOD only: where to start. Formats: `962`, `16:02`, `1:02:03`, `16m2s`, `1h2m3s`. If omitted, the URL's `t=` / `start=` value is used (`…&t=962s`). Ignored for live streams (they always start at the live edge). A start past the end of the video is rejected. |
| `--fast` | off | VOD only: process faster than real time (no 1x pacing). Useful for transcribing a whole video. Can't be combined with `--mpv`. |
| `--preset NAME` | `[translate] preset` in config (`balanced`) | Translation preset: `balanced`, `quality`, `light`, `light-sugoi`, or `none` (Japanese only, no translator loaded). See [Translation presets](#translation-presets). |
| `--mpv` | off (`[mpv] enabled`) | Watch the video in mpv, delayed so subtitles appear exactly when each line is spoken. |
| `--delay SECONDS` | preset's `mpv_delay_s`, else `[mpv] delay_s` (9) | mpv only: how far playback stays behind the pipeline. Raise it if subtitles come late (see [Tuning the delay](#tuning-the-delay)). |
| `--no-web` | web on | Don't start the browser viewer. |
| `--config FILE` | `config.toml` | Use another config file. Missing keys fall back to built-in defaults; a missing file means all defaults. |
| `-v`, `--verbose` | off | Debug logging (ffmpeg / yt-dlp details, per-stage events). |

Examples:

```powershell
uv run auto-translate "https://youtu.be/VIDEO_ID?t=3725"               # VOD from 1:02:05
uv run auto-translate "https://youtu.be/VIDEO_ID" --start 1h2m5s       # same
uv run auto-translate "https://youtu.be/VIDEO_ID" --preset light-sugoi # fast NMT instead of the LLM
uv run auto-translate "https://youtu.be/VIDEO_ID" --mode live --no-web # skip the live check, terminal only
uv run auto-translate "https://youtu.be/VIDEO_ID" --config my.toml -v
```

## Live vs VOD

| | Live | VOD / local file |
|---|---|---|
| Where it starts | The live edge (now) | `--start`, the URL's `t=`, or 0:00 |
| Speed | Real time (audio arrives at 1x) | 1x by default; `--fast` = as fast as possible |
| Timestamps | Seconds since the app started | Position in the video (matches the YouTube player) |
| mpv start | After `delay_s` (9 s) | After `start_wait_s` (3 s) |
| mpv jump keys | No ("VOD only" note) | Yes (←/→ 30 s, Shift+←/→ 5 min) |

## Translation presets

Chosen with `--preset` or `[translate] preset` in `config.toml`.

| Preset | Translator | Runs on | Speed / quality | Setup |
|---|---|---|---|---|
| `balanced` (default) | Ollama `qwen3:4b-instruct-2507-q4_K_M` | GPU (~3 GB VRAM) | ~0.5 s/line; natural, uses the previous lines as context | `ollama pull qwen3:4b-instruct-2507-q4_K_M` |
| `quality` | Ollama `qwen3.5:9b` | GPU + CPU (43% / 57%, doesn't fit in 6 GB) | ~2 s/line (max ~5 s); best wording. mpv delay 15 s. | `ollama pull qwen3.5:9b` |
| `light-sugoi` | Sugoi v4 NMT (CTranslate2) | GPU (~300 MB) | ~45 ms/line; good for casual/game speech. **License: research use only.** | None (auto-download) |
| `light` | Opus-MT ja-en (CTranslate2) | CPU | Fast; stiff, sometimes drops content | None (auto-download) |
| `none` | None | n/a | Japanese only | None |

LLM presets get the previous `context_lines` (4) JA/EN lines and the matching glossary entries in the
prompt, so they handle pronouns and running jokes better. NMT presets translate each line on its own.

You can add your own presets in `config.toml`; any OpenAI-compatible server works (Ollama, llama.cpp
`llama-server`, LM Studio, …):

```toml
[translate.presets.my-llama]
backend = "openai-compat"
base_url = "http://127.0.0.1:8080/v1"   # llama-server
model = "whatever-name"
temperature = 0.2
max_tokens = 120
timeout_s = 20.0
load_timeout_s = 300.0
extra_body = { }                        # extra JSON fields for the request
mpv_delay_s = 12.0                      # optional: mpv delay when this preset is used

[translate.presets.my-nmt]
backend = "ct2"
model = "some-user/some-ctranslate2-model"   # HF repo or local folder
source_spm = "source.spm"                    # tokenizer paths inside the model folder
target_spm = "target.spm"
device = "cpu"                               # or "cuda"
compute_type = "int8"
beam_size = 4
```

Then: `uv run auto-translate URL --preset my-llama`.

> Use `127.0.0.1`, not `localhost`, in `base_url`: on Windows `localhost` tries IPv6 first and adds ~2 s
> to every request.

## Outputs

### Terminal

```
[16:02.4–16:05.1] 今日めっちゃ眠いわ    (vad 0.52s · wait 0.00s · asr 0.31s · mt 0.48s)
                  → I'm so sleepy today.
```

- `[start–end]`: position in the video (VOD) or time since start (live).
- `vad`: time from the end of the audio to the line being cut (higher for long lines that were split).
- `wait`: time waiting for the ASR (should be ~0; higher means the GPU is falling behind).
- `asr` / `mt`: transcription / translation time for this line.

Turn it off with `[output] terminal = false`.

### Browser viewer

Starts by default on port **8765**, on all network interfaces, so other devices on your LAN can open it.

- **This PC:** http://127.0.0.1:8765/
- **Phone / Mac:** `http://<PC's LAN IP>:8765/` (printed at startup).
- Buttons: **JA** (show/hide Japanese), **A− / A+** (text size). Choices are remembered in the browser.
- `?lines=5` in the URL shows more lines (default 3).
- If the connection drops (phone Wi-Fi), the page reconnects by itself and gets the missed lines.
- The browser viewer is **not delayed**: lines appear ~1–3 s after they're spoken on the live stream.
  Use mpv for perfect sync.

LAN access needs Windows Firewall to allow `python.exe` on **Private** networks (Windows asks the first
time). Set `web_host = "127.0.0.1"` to keep it on this PC only. Disable with `--no-web` or `[output] web = false`.

### mpv

See the next section.

## mpv: synced video + subtitles

```powershell
uv run auto-translate "https://www.youtube.com/watch?v=VIDEO_ID" --mpv
```

The app opens mpv, downloads the video (≤480p) along with the audio, and plays it **`delay_s` (9 s; 15 s
with `quality`) behind**. That head start lets each line be transcribed and translated before it's spoken in the video,
so subtitles appear exactly with the speech (measured: within ~0.1 s). English is drawn large, Japanese small above it.

- **Live:** playback starts after 9 s, and stays 9 s behind the real live stream.
- **VOD:** playback starts after ~3 s (the first 9 s of audio are read at full speed).
- Closing the mpv window stops the app.

### mpv keys

mpv's default keys are replaced; seeking with mpv itself would break the incoming stream.

| Key | Action |
|---|---|
| ← / → | VOD: jump back / forward 30 s (~5 s to resume) |
| Shift+← / Shift+→ | VOD: jump back / forward 5 min |
| Space, `p` | Pause / resume |
| `m` | Mute |
| ↑ / ↓, mouse wheel, `9` / `0` | Volume |
| `f`, double-click | Fullscreen (Esc leaves it) |
| `q` | Quit (stops the app) |

### Tuning the delay

The delay must be larger than the longest line (`[vad] max_utterance_s`) plus the time from end of speech
to a finished subtitle (the `latency ... max` in the stats line). Otherwise late lines show up after
they were spoken.

| Preset | Latency max (measured) | Delay used |
|---|---|---|
| `balanced` | ~2 s | 9 s (`[mpv] delay_s`) |
| `quality` | ~8 s | 15 s (preset's `mpv_delay_s`) |

Where the delay comes from, highest priority first:

1. `--delay SECONDS` on the command line.
2. `mpv_delay_s` in the active preset (`[translate.presets.NAME]`).
3. `[mpv] delay_s` (9 s).

A lower delay feels closer to live but risks late subtitles. For VODs the delay doesn't slow the start
(still ~3 s); for live streams the video simply runs that many seconds behind the real stream.

At the end the log prints a sync summary, e.g.
`mpv sync: 14 lines shown, 14 within 0.5s of speech, avg +0.05s, worst +0.13s`.

## Glossary

`glossary.toml` holds fixed JA → EN translations for slang, names and terms ASR or the translator gets wrong:

```toml
[terms]
"草" = "lol"
"てぇてぇ" = "so precious"
"スパチャ" = "superchat"
"ヴァロ" = "Valorant"
```

- **LLM presets:** matching entries are added to the prompt ("use these translations").
- **NMT presets:** the Japanese term is replaced by the English word before translating.
- Longer terms win over shorter ones. Avoid single-kanji terms when possible (they match inside other words).
- Useful for streamer names, game names, and ASR mishearings (add the misheard form too, e.g. `"バロ" = "Valorant"`).

Use another file with `[translate] glossary = "path/to/file.toml"`.

## Configuration reference (`config.toml`)

All keys are optional; missing ones use the defaults below. An unknown key is an error (typo protection).

### Top level

| Key | Default | Description |
|---|---|---|
| `stats_interval_s` | `30` | How often the stats line (queues + latency) is logged. |

### `[ingest]`

| Key | Default | Description |
|---|---|---|
| `ytdlp_format` | `"bestaudio/93/best"` | yt-dlp format for the audio (without mpv). Live streams often have no audio-only format, so `93` (360p) is the fallback. |
| `cookies_from_browser` | `""` | `"chrome"`, `"firefox"`, `"edge"`, … to use your browser's YouTube login. Needed for **members-only** streams. |

### `[vad]`: cutting speech into lines

| Key | Default | Description |
|---|---|---|
| `threshold` | `0.5` | Speech probability that starts a line. Raise if music/noise triggers lines. |
| `neg_threshold` | `0.35` | Below this counts as silence. |
| `min_silence_ms` | `500` | Pause length that ends a line. Lower = shorter lines and lower latency; higher = fuller sentences. |
| `speech_pad_ms` | `200` | Audio kept before/after the speech. |
| `min_speech_ms` | `250` | Drop lines with less speech than this (coughs, clicks). |
| `max_utterance_s` | `6.0` (code default 9) | Force-split long speech after this many seconds, at the quietest point in the last 2 s. Keep below `[mpv] delay_s` − 3. |

### `[asr]`: speech recognition

| Key | Default | Description |
|---|---|---|
| `backend` | `"faster-whisper"` | Only backend for now. |
| `model` | `"kotoba-tech/kotoba-whisper-v2.0-faster"` | HF repo id or local folder (CTranslate2 format). `"large-v3"` also works (slower, more VRAM). |
| `device` | `"cuda"` | `"cuda"` or `"cpu"`. |
| `compute_type` | `"int8_float16"` | ~1.3 GB VRAM. `"float16"` uses ~1 GB more for the same accuracy. On CPU use `"int8"`. |
| `beam_size` | `5` | 1 = fastest; 5 = more accurate. |
| `no_speech_threshold` | `0.6` | A segment is dropped when no-speech probability is above this **and**… |
| `log_prob_threshold` | `-1.0` | …the average log-probability is below this (Whisper's hallucination filter). |
| `blocklist` | see config | Lines containing any of these are dropped (known Whisper hallucinations on silence/music, e.g. 「ご視聴ありがとうございました」). |

### `[translate]`

| Key | Default | Description |
|---|---|---|
| `preset` | `"balanced"` | Preset name (overridden by `--preset`). `"none"` = no translation. |
| `context_lines` | `4` | Previous JA/EN lines given to the LLM as context. |
| `glossary` | `"glossary.toml"` | Glossary file. |

`[translate.presets.NAME]` tables define presets (see [Translation presets](#translation-presets)).
Keys for `backend = "openai-compat"`: `model`, `base_url`, `temperature` (0.2), `max_tokens` (120),
`timeout_s` (20), `load_timeout_s` (300, first request that loads the model), `extra_body` (merged into the
request, e.g. `{ reasoning_effort = "none" }` to stop a model from "thinking"), `mpv_delay_s` (optional, mpv delay for this preset).
Keys for `backend = "ct2"`: `model`, `source_spm`, `target_spm`, `device` (`cpu`), `compute_type` (`int8`), `beam_size` (4).

### `[output]`

| Key | Default | Description |
|---|---|---|
| `terminal` | `true` | Print lines to the terminal. |
| `web` | `true` | Run the browser viewer (`--no-web` turns it off). |
| `web_host` | `"0.0.0.0"` | `0.0.0.0` = reachable from the LAN; `127.0.0.1` = this PC only. |
| `web_port` | `8765` | Viewer port. |

### `[mpv]`

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Always use mpv (same as `--mpv`). |
| `path` | `""` | mpv executable. `""` = `PATH`, then `C:\Program Files\MPV Player\mpv.exe`. |
| `delay_s` | `9.0` | How far playback stays behind the pipeline. Must exceed `max_utterance_s` + max latency. A preset's `mpv_delay_s` or `--delay` overrides it. |
| `start_wait_s` | `3.0` | VOD: wait before playing after start or a jump. |
| `port` | `9137` | Local TCP port ffmpeg streams the video to mpv on. Change if it's in use. |
| `hwdec` | `"auto-safe"` | mpv video decoding. `"auto-safe"` = GPU video decoder (NVDEC / D3D11VA), so playback stays smooth while the LLM uses the CPU. `"no"` = CPU decoding. |
| `ytdlp_format` | ≤480p H.264 HLS | yt-dlp format for the video. Raise `height<=480` for better picture (more bandwidth/decoding). |
| `en_size` / `ja_size` | `44` / `28` | Subtitle font sizes (on a 720-line canvas). |
| `min_display_s` | `1.5` | Show short lines at least this long. |
| `hold_s` | `1.0` | Keep a line on screen this long after it ends (unless the next line starts). |

## Reading the logs

Every `stats_interval_s` the app logs something like:

```
12:00:30 INFO  auto_translate.pipeline: [30s] queues {...} | 12 lines · asr avg 0.33s max 0.52s · mt avg 0.49s max 0.81s · latency avg 1.31s max 2.10s · max queue ...
```

- **queues** should stay at 0–1. A growing queue (and a `falling behind` warning) means the GPU can't keep
  up: use a lighter preset, `beam_size = 1`, or close other GPU apps.
- **latency**: end of speech → subtitle ready. ~1–1.5 s with `balanced`.
- A summary is printed when the app stops.

## Helper scripts

| Script | Use |
|---|---|
| `uv run python scripts/m0_cuda_test.py clip.wav [--model ID] [--compute-type float16]` | Check that Whisper runs on the GPU; prints the text and real-time factor. |
| `uv run python scripts/m2_compare.py scripts/sample_ja.txt light light-sugoi balanced quality` | Translate the same Japanese lines with several presets and compare output and speed. |
| `uv run python scripts/mpv_jump_probe.py [RIGHT LEFT ...]` | Debug: while `--mpv` plays a VOD, press jump keys and log the audio state after each jump. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HTTP Error 404: Not Found` at startup (LLM preset) | The model isn't pulled in Ollama. Run `ollama list`, then `ollama pull <model>` from the preset. |
| `Connection refused` at startup (LLM preset) | Ollama isn't running. Start the Ollama app (or `ollama serve`). |
| First start hangs ~1–2 min after "ASR ready" | Normal: Ollama is loading the LLM into VRAM. |
| yt-dlp errors / "Sign in to confirm" / format not available | Update yt-dlp. For members-only streams set `cookies_from_browser`. |
| Phone can't open the viewer | Same Wi-Fi? Allow `python.exe` on Private networks in Windows Firewall; check the network is set to "Private". |
| Subtitles late in mpv | Raise the delay (`--delay 15`, or `mpv_delay_s` in the preset), lower `[vad] max_utterance_s`, or use a faster preset. |
| mpv video stutters / drops frames (esp. with `quality`) | Keep `[mpv] hwdec = "auto-safe"` (GPU decoding). Check Task Manager → Performance → GPU → "Video Decode" is active while playing. Otherwise use 30 fps: add `[fps<=30]` to `[mpv] ytdlp_format`. |
| `--mpv` can't be combined with `--fast` | mpv plays at 1x; drop one of them. |
| Out of VRAM / CUDA errors | Close other GPU apps; use `compute_type = "int8_float16"`; use `light`/`light-sugoi` instead of an LLM preset. |
| Odd lines like 「ご視聴ありがとうございました」 during music | Whisper hallucination; add the phrase to `[asr] blocklist`. |
| Names / game titles wrong | Add them (and their misheard forms) to `glossary.toml`. |
| Very slow everything | Laptop on battery? Plug it in. |
