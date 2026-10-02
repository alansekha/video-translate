from ..config import Ct2NmtConfig, OpenAICompatConfig, TranslateConfig
from ..glossary import Glossary
from .base import Translator


def create_translator(cfg: TranslateConfig, glossary: Glossary) -> Translator | None:
    """Build the translator for cfg.preset; None means translation is off."""
    if cfg.preset == "none":
        return None
    if cfg.preset not in cfg.presets:
        raise SystemExit(f"unknown preset {cfg.preset!r}; config.toml has: {', '.join(cfg.presets)}")
    settings = dict(cfg.presets[cfg.preset])  # copy, so pop() doesn't change the config
    backend = settings.pop("backend")
    settings.pop("mpv_delay_s", None)  # read by __main__ (mpv delay for slow presets), not a backend setting
    # Imports inside the branches: an unused backend's libraries are never loaded.
    if backend == "ct2":
        from .ct2_nmt import Ct2NmtTranslator

        return Ct2NmtTranslator(Ct2NmtConfig(**settings), glossary)
    if backend == "openai-compat":
        from .openai_compat import OpenAICompatTranslator

        return OpenAICompatTranslator(OpenAICompatConfig(**settings), glossary)
    raise SystemExit(f"unknown translate backend {backend!r} in preset {cfg.preset!r}")
