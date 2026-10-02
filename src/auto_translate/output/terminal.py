from ..types import Subtitle


def fmt_ts(seconds: float) -> str:
    """12.4 → '00:12.4', 3725.0 → '1:02:05.0'."""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m:02d}:{s:04.1f}"


class TerminalOutput:
    def show(self, sub: Subtitle) -> None:
        print_subtitle(sub)

    def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


def print_subtitle(sub: Subtitle) -> None:
    tm = sub.timings
    lat = (f"vad {tm.segmented - tm.audio_read:.2f}s · wait {tm.asr_start - tm.segmented:.2f}s · "
           f"asr {tm.asr_end - tm.asr_start:.2f}s · mt {tm.mt_end - tm.mt_start:.2f}s")
    stamp = f"[{fmt_ts(sub.start_s)}–{fmt_ts(sub.end_s)}]"
    print(f"{stamp} {sub.ja}    ({lat})", flush=True)
    if sub.en:
        print(f"{' ' * len(stamp)} → {sub.en}", flush=True)
