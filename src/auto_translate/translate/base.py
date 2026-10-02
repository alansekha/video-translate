from typing import Protocol


class Translator(Protocol):
    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        """Translate one line. `history` = previous (ja, en) pairs, oldest first (may be ignored)."""
        ...
