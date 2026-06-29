"""Full Lumen-1 multimodal model."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.model.audio_encoder import AudioEncoder
from lumen.model.config import LumenConfig, ModelSize
from lumen.model.decoder import LumenDecoder
from lumen.model.projectors import ModalityProjector
from lumen.model.safety_head import SafetyHead
from lumen.model.vision_encoder import VisionEncoder


@dataclass
class LumenOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    safety_logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    past_key_values: Optional[list] = None


class LumenModel(nn.Module):
    """Multimodal backbone: encoders + decoder."""

    def __init__(self, config: LumenConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.vision_projector = ModalityProjector(
            config.vision_hidden_size,
            config.projector_hidden_size,
            config.hidden_size,
        )
        self.audio_projector = ModalityProjector(
            config.audio_hidden_size,
            config.projector_hidden_size,
            config.hidden_size,
        )
        self.decoder = LumenDecoder(config)

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(pixel_values)
        return self.vision_projector(features)

    def encode_audio(self, mel: torch.Tensor) -> torch.Tensor:
        features = self.audio_encoder(mel)
        return self.audio_projector(features)

    def build_multimodal_embeds(
        self,
        input_ids: torch.Tensor,
        vision_embeds: Optional[List[torch.Tensor]] = None,
        audio_embeds: Optional[List[torch.Tensor]] = None,
        vision_token_id: int = -1,
        audio_token_id: int = -1,
    ) -> torch.Tensor:
        """
        Replace placeholder modality token positions with continuous embeddings.
        Placeholder tokens in input_ids mark where vision/audio spans go.
        """
        text_embeds = self.decoder.embed_tokens(input_ids).clone()
        batch_size = input_ids.shape[0]

        if vision_embeds is None and audio_embeds is None:
            return text_embeds

        for b in range(batch_size):
            if vision_embeds and vision_embeds[b] is not None:
                v_positions = (input_ids[b] == vision_token_id).nonzero(as_tuple=True)[0]
                v_len = min(len(v_positions), vision_embeds[b].shape[0])
                if v_len > 0:
                    text_embeds[b, v_positions[:v_len]] = vision_embeds[b][:v_len]

            if audio_embeds and audio_embeds[b] is not None:
                a_positions = (input_ids[b] == audio_token_id).nonzero(as_tuple=True)[0]
                a_len = min(len(a_positions), audio_embeds[b].shape[0])
                if a_len > 0:
                    text_embeds[b, a_positions[:a_len]] = audio_embeds[b][:a_len]

        return text_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        mel_spectrograms: Optional[torch.Tensor] = None,
        vision_placeholder_id: int = -1,
        audio_placeholder_id: int = -1,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> LumenOutput:
        vision_embeds_list = None
        audio_embeds_list = None

        if pixel_values is not None:
            if pixel_values.dim() == 5:
                # Encode video frames and project to decoder hidden dimension
                # (video frames encode කර projector එක හරහා decoder dimension එකට project කරනවා)
                vision_feats = self.vision_encoder.encode_video(pixel_values)
                vision_feats = self.vision_projector(vision_feats)
            else:
                # Encode single image/photo and project to decoder hidden dimension
                # (තනි රූපයක්/ඡායාරූපයක් encode කර projector එක හරහා project කරනවා)
                vision_feats = self.encode_vision(pixel_values)
            vision_embeds_list = [vision_feats[i] for i in range(vision_feats.shape[0])]

        if mel_spectrograms is not None:
            audio_feats = self.encode_audio(mel_spectrograms)
            audio_embeds_list = [audio_feats[i] for i in range(audio_feats.shape[0])]

        inputs_embeds = self.build_multimodal_embeds(
            input_ids,
            vision_embeds_list,
            audio_embeds_list,
            vision_token_id=vision_placeholder_id,
            audio_token_id=audio_placeholder_id,
        )

        hidden_states, presents = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return LumenOutput(
            hidden_states=hidden_states,
            past_key_values=presents,
        )


class LumenForCausalLM(nn.Module):
    """Lumen-1 with LM head and safety head."""

    def __init__(self, config: LumenConfig):
        super().__init__()
        self.config = config
        self.model = LumenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.safety_head = SafetyHead(config)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.decoder.embed_tokens.weight

    @classmethod
    def from_size(cls, size: Union[ModelSize, str]) -> "LumenForCausalLM":
        if isinstance(size, str):
            size = ModelSize(size)
        config = LumenConfig.from_size(size)
        return cls(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.decoder.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        mel_spectrograms: Optional[torch.Tensor] = None,
        vision_placeholder_id: int = -1,
        audio_placeholder_id: int = -1,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        return_safety_logits: bool = True,
    ) -> LumenOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            mel_spectrograms=mel_spectrograms,
            vision_placeholder_id=vision_placeholder_id,
            audio_placeholder_id=audio_placeholder_id,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
        )

        hidden_states = outputs.hidden_states
        logits = self.lm_head(hidden_states)
        safety_logits = self.safety_head(hidden_states) if return_safety_logits else None

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return LumenOutput(
            loss=loss,
            logits=logits,
            safety_logits=safety_logits,
            hidden_states=hidden_states,
            past_key_values=outputs.past_key_values,
        )

    def count_parameters(self) -> Dict[str, int]:
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "decoder": _count(self.model.decoder),
            "vision_encoder": _count(self.model.vision_encoder),
            "audio_encoder": _count(self.model.audio_encoder),
            "vision_projector": _count(self.model.vision_projector),
            "audio_projector": _count(self.model.audio_projector),
            "safety_head": _count(self.safety_head),
            "total": _count(self),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        self.eval()
        generated = input_ids
        past_key_values = None

        for step in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_safety_logits=False,
                **kwargs,
            )
            past_key_values = outputs.past_key_values
            
            # After the first step, clear pixel_values and mel_spectrograms to avoid re-running heavy encoders
            # (පළමු පියවරෙන් පසු, encoders නැවත ක්‍රියාත්මක වීම වැළැක්වීමට pixel_values සහ mel_spectrograms ඉවත් කරමු)
            if step == 0:
                kwargs.pop("pixel_values", None)
                kwargs.pop("mel_spectrograms", None)
            next_logits = outputs.logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                score = torch.gather(next_logits, 1, generated)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                next_logits.scatter_(1, generated, score)

            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = float("-inf")
                next_logits = torch.zeros_like(next_logits).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
