from typing import Protocol

from ..types import Subtitle


class Output(Protocol):
    def show(self, sub: Subtitle) -> None: ...

    def drain(self) -> None:
        """The stream ended normally: finish anything still pending (mpv plays out its buffer)."""
        ...

    def close(self) -> None: ...
