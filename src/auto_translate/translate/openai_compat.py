"""LLM backend over an OpenAI-style chat API (Ollama, llama.cpp llama-server)."""

import json
import urllib.request

from ..config import OpenAICompatConfig
from ..glossary import Glossary

SYSTEM_PROMPT = """\
You translate Japanese livestream speech (VTubers, gamers) into English subtitles.
The Japanese comes from speech recognition: it is casual, may lack punctuation, and may contain mistakes.
Rules:
- Translate ONLY the line after "Translate:". Earlier lines are context.
- Output only the English subtitle: one short, natural, casual line. No quotes, notes or romaji.
- Use the glossary translations when those terms appear.
- If the line is just filler or a noise ("あー", "えっと"), output a short equivalent ("Ah.", "Um.")."""


class OpenAICompatTranslator:
    def __init__(self, cfg: OpenAICompatConfig, glossary: Glossary) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        # Warm-up: makes the server load the model now, not on the first real subtitle line.
        self._chat("Translate:\nJA: こんにちは", timeout=cfg.load_timeout_s)

    def build_prompt(self, ja: str, history: list[tuple[str, str]]) -> str:
        parts = []
        if hits := self.glossary.find(ja):
            parts.append("Glossary:\n" + "\n".join(f"{term} = {en}" for term, en in hits))
        if history:
            parts.append("Context:\n" + "\n".join(f"JA: {j}\nEN: {e}" for j, e in history))
        parts.append(f"Translate:\nJA: {ja}")
        return "\n\n".join(parts)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        return _clean(self._chat(self.build_prompt(ja, history), timeout=self.cfg.timeout_s))

    def _chat(self, user_msg: str, timeout: float) -> str:
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
            **self.cfg.extra_body,
        }
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]


def _clean(text: str) -> str:
    """Keep the first non-empty line, minus an 'EN:' prefix and surrounding quotes."""
    line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
    if line.upper().startswith("EN:"):
        line = line[3:].strip()
    return line.strip('"“”「」')
