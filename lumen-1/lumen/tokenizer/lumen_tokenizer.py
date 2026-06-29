"""LumenTokenizer: 128K BPE with multimodal and safety special tokens."""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Union

# Reserved special tokens (~512 slots; core set defined here)
SPECIAL_TOKENS: Dict[str, str] = {
    # Modality delimiters
    "<image>": "<image>",
    "</image>": "</image>",
    "<audio>": "<audio>",
    "</audio>": "</audio>",
    "<video>": "<video>",
    "</video>": "</video>",
    "<memory>": "<memory>",
    "</memory>": "</memory>",
    # Chat control
    "<|system|>": "<|system|>",
    "<|user|>": "<|user|>",
    "<|assistant|>": "<|assistant|>",
    "<|tool|>": "<|tool|>",
    "<|endoftext|>": "<|endoftext|>",
    # Safety
    "<|refuse|>": "<|refuse|>",
    "<|uncertain|>": "<|uncertain|>",
    # Modality placeholders (replaced by continuous embeddings at runtime)
    "<|vision_pad|>": "<|vision_pad|>",
    "<|audio_pad|>": "<|audio_pad|>",
}

# Code-aware pretokenization pattern (tiktoken cl100k style)
CODE_PATTERN = re.compile(
    r"''|'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """NFKC normalization with whitespace cleanup."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class LumenTokenizer:
    """
    SentencePiece BPE tokenizer with byte-level fallback and special tokens.

    Training: run `train_tokenizer()` on a corpus to produce the .model file.
    Inference: load from pretrained path.
    """

    VOCAB_SIZE = 128256
    PAD_TOKEN = "<|endoftext|>"
    BOS_TOKEN = "<|endoftext|>"
    EOS_TOKEN = "<|endoftext|>"

    def __init__(self, model_path: Optional[str] = None):
        self.sp = None
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._init_untrained()

    def _init_untrained(self) -> None:
        """Initialize token maps with special tokens only (pre-training state)."""
        for i, token in enumerate(SPECIAL_TOKENS.values()):
            self._token_to_id[token] = i
            self._id_to_token[i] = token

    def load(self, model_path: str) -> None:
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self._rebuild_maps()

    def _rebuild_maps(self) -> None:
        if self.sp is None:
            return
        self._token_to_id = {self.sp.IdToPiece(i): i for i in range(self.sp.GetPieceSize())}
        self._id_to_token = {i: self.sp.IdToPiece(i) for i in range(self.sp.GetPieceSize())}

    @property
    def vocab_size(self) -> int:
        if self.sp:
            return self.sp.GetPieceSize()
        return self.VOCAB_SIZE

    @property
    def vision_pad_id(self) -> int:
        return self.token_to_id("<|vision_pad|>")

    @property
    def audio_pad_id(self) -> int:
        return self.token_to_id("<|audio_pad|>")

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if self.sp:
            return self.sp.PieceToId(token)
        raise KeyError(f"Unknown token: {token}")

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        if self.sp:
            return self.sp.IdToPiece(idx)
        raise KeyError(f"Unknown id: {idx}")

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        text = normalize_text(text)
        if self.sp:
            ids = self.sp.EncodeAsIds(text)
        else:
            ids = [ord(c) % 256 + len(SPECIAL_TOKENS) for c in text]

        if add_bos:
            ids = [self.token_to_id(self.BOS_TOKEN)] + ids
        if add_eos:
            ids = ids + [self.token_to_id(self.EOS_TOKEN)]
        return ids

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        if self.sp:
            tokens = [self.id_to_token(i) for i in ids]
            if skip_special:
                special = set(SPECIAL_TOKENS.values())
                tokens = [t for t in tokens if t not in special]
            return self.sp.DecodeIds(ids) if not skip_special else "".join(tokens)
        return "".join(chr((i - len(SPECIAL_TOKENS)) % 256) for i in ids)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages into Lumen chat template."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
            elif role == "tool":
                parts.append(f"<|tool|>\n{content}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def build_multimodal_sequence(
        self,
        text_parts: Union[str, List[str]],
        num_vision_tokens: int = 0,
        num_audio_tokens: int = 0,
    ) -> List[int]:
        """Build token sequence with vision/audio placeholder pads inserted at exact tag locations."""
        # Convert list of parts to a single flat string if needed
        # (ලැබෙන text_parts ලැයිස්තුව තනි string එකක් බවට පරිවර්තනය කරගන්නවා)
        if isinstance(text_parts, list):
            text = "".join(text_parts)
        else:
            text = text_parts
            
        # Split text on <image> and <audio> tags to insert pads in-place
        # (රූප සහ ශබ්ද ඛණ්ඩ නියමිත ස්ථාන වලට ඇතුළත් කිරීමට tags අනුව split කරගන්නවා)
        tokens = re.split(r"(<image>|<audio>)", text)
        ids = []
        for token in tokens:
            if token == "<image>":
                if num_vision_tokens > 0:
                    ids.extend([self.vision_pad_id] * num_vision_tokens)
            elif token == "<audio>":
                if num_audio_tokens > 0:
                    ids.extend([self.audio_pad_id] * num_audio_tokens)
            else:
                if token:
                    ids.extend(self.encode(token))
        return ids

    def save_config(self, path: Union[str, Path]) -> None:
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "bos_token": self.BOS_TOKEN,
            "eos_token": self.EOS_TOKEN,
            "pad_token": self.PAD_TOKEN,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def train(
        cls,
        corpus_paths: List[str],
        output_dir: str,
        vocab_size: int = 128256,
        character_coverage: float = 0.9995,
    ) -> "LumenTokenizer":
        """
        Train SentencePiece BPE model on corpus files.

        Args:
            corpus_paths: List of plain-text training files
            output_dir: Directory to write lumen_tokenizer.model
        """
        os.makedirs(output_dir, exist_ok=True)
        model_prefix = os.path.join(output_dir, "lumen_tokenizer")

        # Write user-defined symbols for special tokens
        user_symbols = list(SPECIAL_TOKENS.values())

        import sentencepiece as spm

        spm.SentencePieceTrainer.train(
            input=",".join(corpus_paths),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type="bpe",
            byte_fallback=True,
            normalization_rule_name="nmt_nfkc",
            user_defined_symbols=user_symbols,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            train_extremely_large_corpus=True,
        )

        tokenizer = cls(model_path=f"{model_prefix}.model")
        tokenizer.save_config(os.path.join(output_dir, "tokenizer_config.json"))
        return tokenizer
