"""Speculative decoding with Lumen-1-1B draft model."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from lumen.model.lumen_model import LumenForCausalLM


class SpeculativeDecoder:
    """
    Draft-verify speculative decoding.
    Target: 7B model, Draft: 1B model, acceptance rate target 75%+.
    """

    def __init__(
        self,
        target_model: LumenForCausalLM,
        draft_model: LumenForCausalLM,
        gamma: int = 5,
        device: Optional[str] = None,
    ):
        self.target = target_model
        self.draft = draft_model
        self.gamma = gamma
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target.to(self.device).eval()
        self.draft.to(self.device).eval()
        self.stats = {"accepted": 0, "drafted": 0}

    @torch.no_grad()
    def _draft_tokens(
        self,
        input_ids: torch.Tensor,
        gamma: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We always pass the full context sequence draft_ids to preserve attention history (මුළු sequence එකම pass කරමු)
        draft_ids = input_ids
        draft_probs = []
        draft_dists = []
        for _ in range(gamma):
            out = self.draft(draft_ids, **kwargs)
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.argmax(probs, dim=-1, keepdim=True)
            draft_ids = torch.cat([draft_ids, next_tok], dim=-1)
            draft_probs.append(probs.gather(1, next_tok))
            draft_dists.append(probs) # Save the full distribution vector (මුළු vector එකම සේව් කරගන්නවා)
        draft_probs = torch.cat(draft_probs, dim=1)
        draft_dists = torch.cat(draft_dists, dim=0) # shape: (gamma, VocabSize)
        return draft_ids[:, input_ids.shape[1] :], draft_probs, draft_dists

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        draft_tokens, draft_probs, draft_dists = self._draft_tokens(input_ids, self.gamma, **kwargs)
        if draft_tokens.shape[1] == 0:
            out = self.target(input_ids, **kwargs)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            return torch.cat([input_ids, next_tok], dim=-1)

        # Verify with target model
        candidate = torch.cat([input_ids, draft_tokens], dim=-1)
        target_out = self.target(candidate, **kwargs)
        target_logits = target_out.logits[:, input_ids.shape[1] - 1 : -1, :]
        target_probs = F.softmax(target_logits, dim=-1)
        target_token_probs = target_probs.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)

        # Accept tokens where target agrees
        accepted = []
        for i in range(draft_tokens.shape[1]):
            self.stats["drafted"] += 1
            if target_token_probs[0, i] >= draft_probs[0, i]:
                accepted.append(draft_tokens[0, i].item())
                self.stats["accepted"] += 1
            else:
                # Resample from adjusted distribution: p'_adj(x) = max(0, p(x) - q(x))
                # (මෙහිදී draft_probs[0, i] වෙනුවට සම්පූර්ණ draft distribution vector එක වන draft_dists[i] භාවිතා කරයි)
                adjusted = torch.clamp(target_probs[0, i] - draft_dists[i], min=0)
                if adjusted.sum() > 0:
                    adjusted = adjusted / adjusted.sum()
                    resampled = torch.multinomial(adjusted, 1).item()
                    accepted.append(resampled)
                break

        if not accepted:
            out = self.target(input_ids, **kwargs)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            accepted = [next_tok]

        new_ids = torch.cat([input_ids, torch.tensor([accepted], device=input_ids.device)], dim=-1)
        return new_ids

    @property
    def acceptance_rate(self) -> float:
        if self.stats["drafted"] == 0:
            return 0.0
        return self.stats["accepted"] / self.stats["drafted"]
