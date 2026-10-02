"""M2: translate the same Japanese lines with several presets and compare quality + speed.

Usage:
    uv run python scripts/m2_compare.py scripts/sample_ja.txt light light-sugoi balanced
"""

import sys
import time
from collections import deque
from pathlib import Path

from auto_translate import cuda_dlls

cuda_dlls.setup()

from auto_translate.config import load_config  # noqa: E402
from auto_translate.glossary import Glossary  # noqa: E402
from auto_translate.translate import create_translator  # noqa: E402


def main() -> None:
    lines_file, presets = Path(sys.argv[1]), sys.argv[2:]
    lines = [ln.strip() for ln in lines_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cfg = load_config(Path("config.toml"))
    glossary = Glossary.load(Path(cfg.translate.glossary))

    results: dict[str, list[tuple[str, float]]] = {}
    for preset in presets:
        cfg.translate.preset = preset
        t0 = time.perf_counter()
        translator = create_translator(cfg.translate, glossary)
        assert translator is not None
        print(f"[{preset}] loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        translator.translate("こんにちは", [])  # warm-up (first call is slower)

        history: deque[tuple[str, str]] = deque(maxlen=cfg.translate.context_lines)
        out = []
        for ja in lines:
            t0 = time.perf_counter()
            en = translator.translate(ja, list(history))
            out.append((en, time.perf_counter() - t0))
            history.append((ja, en))
        results[preset] = out

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    width = max(len(p) for p in presets)
    for i, ja in enumerate(lines):
        print(f"\nJA  {ja}")
        for preset in presets:
            en, secs = results[preset][i]
            print(f"  {preset:<{width}} {secs * 1000:5.0f} ms  {en}")

    print()
    for preset in presets:
        times = [s for _, s in results[preset]]
        print(f"{preset:<{width}} avg {sum(times) / len(times) * 1000:.0f} ms, max {max(times) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
