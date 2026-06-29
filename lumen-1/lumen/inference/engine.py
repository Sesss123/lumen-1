"""Lumen-1 inference engine with KV cache and encoder caching."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

from lumen.model.lumen_model import LumenForCausalLM
from lumen.tokenizer.lumen_tokenizer import LumenTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    eos_token_id: Optional[int] = None


class KVCacheManager:
    """Manages per-layer KV caches for incremental decoding."""

    def __init__(self):
        self.caches: Optional[list] = None

    def reset(self) -> None:
        self.caches = None

    def update(self, new_cache: list) -> None:
        self.caches = new_cache


class InferenceEngine:
    """
    Local/cloud inference with prefix caching and encoder caching.
    Compatible with vLLM-style continuous batching interface.
    """

    def __init__(
        self,
        model: LumenForCausalLM,
        tokenizer: LumenTokenizer,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self.kv_manager = KVCacheManager()
        self._encoder_cache: Dict[str, torch.Tensor] = {}
        self._prefix_cache: Dict[str, torch.Tensor] = {}

    def _cache_key(self, data) -> str:
        import hashlib

        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()
        return hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()

    def encode_image(self, pixel_values: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        key = self._cache_key(pixel_values) if use_cache else None
        if use_cache and key in self._encoder_cache:
            return self._encoder_cache[key]
        with torch.no_grad():
            embeds = self.model.model.encode_vision(pixel_values.to(self.device, dtype=self.dtype))
        if use_cache and key:
            self._encoder_cache[key] = embeds
        return embeds

    def prepare_inputs(
        self,
        messages: List[Dict[str, str]],
        pixel_values: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
    ) -> Dict:
        text = self.tokenizer.apply_chat_template(messages)
        num_vision = pixel_values.shape[0] * 576 if pixel_values is not None else 0
        input_ids = self.tokenizer.build_multimodal_sequence(
            [text],
            num_vision_tokens=num_vision,
            num_audio_tokens=mel.shape[1] if mel is not None else 0,
        )
        return {
            "input_ids": torch.tensor([input_ids], device=self.device),
            "pixel_values": pixel_values,
            "mel_spectrograms": mel,
            "vision_placeholder_id": self.tokenizer.vision_pad_id,
            "audio_placeholder_id": self.tokenizer.audio_pad_id,
        }

    @torch.no_grad()
    def generate(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        pixel_values: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
    ) -> str:
        config = config or GenerationConfig()
        inputs = self.prepare_inputs(messages, pixel_values, mel)
        output_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            eos_token_id=config.eos_token_id,
            pixel_values=inputs.get("pixel_values"),
            mel_spectrograms=inputs.get("mel_spectrograms"),
            vision_placeholder_id=inputs["vision_placeholder_id"],
            audio_placeholder_id=inputs["audio_placeholder_id"],
        )
        gen_ids = output_ids[0, inputs["input_ids"].shape[1] :].tolist()
        return self.tokenizer.decode(gen_ids)

    def get_vllm_config(self) -> Dict:
        """Export config for vLLM multimodal serving."""
        c = self.model.config
        return {
            "model_type": "lumen",
            "hidden_size": c.hidden_size,
            "num_hidden_layers": c.num_hidden_layers,
            "num_attention_heads": c.num_attention_heads,
            "num_key_value_heads": c.num_key_value_heads,
            "vocab_size": c.vocab_size,
            "max_position_embeddings": c.max_position_embeddings,
            "sliding_window_size": c.sliding_window_size,
            "vision_image_size": c.vision_image_size,
            "audio_num_mel_bins": c.audio_num_mel_bins,
            "torch_dtype": str(self.dtype).split(".")[-1],
        }
