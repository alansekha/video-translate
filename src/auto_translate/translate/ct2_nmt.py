"""CTranslate2 NMT backend: Opus-MT or Sugoi (any seq2seq model with SentencePiece tokenizers)."""

import json
from pathlib import Path

import ctranslate2
import sentencepiece
from huggingface_hub import snapshot_download

from ..config import Ct2NmtConfig
from ..glossary import Glossary


class Ct2NmtTranslator:
    def __init__(self, cfg: Ct2NmtConfig, glossary: Glossary) -> None:
        self.cfg = cfg
        self.glossary = glossary
        folder = Path(cfg.model) if Path(cfg.model).is_dir() else Path(snapshot_download(cfg.model))
        self.model = ctranslate2.Translator(str(folder), device=cfg.device, compute_type=cfg.compute_type)
        self.src_spm = sentencepiece.SentencePieceProcessor(model_file=str(folder / cfg.source_spm))
        self.tgt_spm = sentencepiece.SentencePieceProcessor(model_file=str(folder / cfg.target_spm))
        # Some converted models add the end-of-sentence token themselves (Sugoi), others expect it (Opus-MT).
        model_cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.append_eos = not model_cfg.get("add_source_eos", False)

    def translate(self, ja: str, history: list[tuple[str, str]]) -> str:
        tokens = self.src_spm.encode(self.glossary.substitute(ja), out_type=str)
        if self.append_eos:
            tokens.append("</s>")
        result = self.model.translate_batch([tokens], beam_size=self.cfg.beam_size,
                                            max_decoding_length=256)
        return self.tgt_spm.decode(result[0].hypotheses[0]).replace("<unk>", "").strip()
