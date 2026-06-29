"""Layer 6 (partial): Refusal handler with structured <|refuse|> responses."""

from typing import Dict, List, Optional


REFUSAL_TEMPLATES = {
    "default": (
        "<|refuse|> I'm not able to help with that request. "
        "I can help with safe, constructive topics instead."
    ),
    "jailbreak": (
        "<|refuse|> I can't override my safety guidelines. "
        "Please ask a different question."
    ),
    "harmful": (
        "<|refuse|> I won't provide information that could cause harm. "
        "If you're in crisis, please contact local emergency services."
    ),
    "pii": (
        "<|refuse|> I detected sensitive personal information in this request. "
        "Please remove PII and try again."
    ),
    "uncertain": (
        "<|uncertain|> I'm not confident enough to answer accurately. "
        "Could you provide more context or rephrase your question?"
    ),
}


class RefusalHandler:
    """Structured refusal responses with safe alternatives."""

    def __init__(self, templates: Optional[Dict[str, str]] = None):
        self.templates = templates or REFUSAL_TEMPLATES

    def refuse(self, reason: str = "default", alternatives: Optional[List[str]] = None) -> str:
        response = self.templates.get(reason, self.templates["default"])
        if alternatives:
            response += "\n\nI can help with:\n" + "\n".join(f"- {a}" for a in alternatives)
        return response

    def should_refuse_safety_head(self, safety_logits, threshold: float = 0.7) -> Optional[str]:
        """Check in-model safety head logits (Layer 3)."""
        import torch

        if safety_logits is None:
            return None
        probs = torch.sigmoid(safety_logits)
        max_prob, idx = probs.max(dim=-1)
        if max_prob.item() > threshold:
            categories = [
                "harmful", "harmful", "harmful", "harmful",
                "harmful", "harmful", "uncertain", "pii",
            ]
            return categories[min(idx.item(), len(categories) - 1)]
        return None
