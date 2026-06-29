"""Layer 1: Prompt processing — normalize, jailbreak detection, schema enforcement."""

import re
from typing import Dict, List, Optional, Tuple

JAILBREAK_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"you are now (DAN|jailbroken|unrestricted)",
    r"pretend you have no (rules|restrictions|guidelines)",
    r"developer mode enabled",
    r"bypass (safety|content) (filter|policy)",
]


class PromptProcessor:
    """Rule engine + lightweight pattern classifier for prompt preprocessing."""

    def __init__(self, max_length: int = 32_768):
        self.max_length = max_length
        self._patterns = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]

    def normalize(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text[: self.max_length]

    def detect_jailbreak(self, text: str) -> Tuple[bool, float]:
        hits = sum(1 for p in self._patterns if p.search(text))
        score = min(1.0, hits * 0.35)
        return hits > 0, score

    def enforce_schema(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensure messages follow system/user/assistant/tool schema."""
        valid_roles = {"system", "user", "assistant", "tool"}
        cleaned = []
        for msg in messages:
            role = msg.get("role", "user")
            if role not in valid_roles:
                role = "user"
            content = self.normalize(str(msg.get("content", "")))
            if content:
                cleaned.append({"role": role, "content": content})
        return cleaned

    def process(self, messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict]:
        messages = self.enforce_schema(messages)
        combined = " ".join(m["content"] for m in messages)
        is_jailbreak, score = self.detect_jailbreak(combined)
        meta = {"jailbreak_detected": is_jailbreak, "jailbreak_score": score}
        return messages, meta
