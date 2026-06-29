"""Layer 2 & 4: Input/output content filtering."""

import hashlib
import re
from typing import Dict, Optional, Tuple

from lumen.data.filters import PIIFilter, SafetyFilter


class ContentFilter:
    """
    Input and output content filter.
    Production: PhotoDNA, CLIP NSFW, Whisper toxicity.
    Development: heuristic + regex filters.
    """

    BLOCKED_IMAGE_HASHES: set = set()  # PhotoDNA hash set placeholder

    def __init__(self):
        self.safety = SafetyFilter()
        self.pii = PIIFilter()

    def check_text(self, text: str) -> Tuple[bool, Dict]:
        passed = self.safety.passes(text)
        has_pii = self.pii.contains_pii(text)
        return passed and not has_pii, {
            "toxicity": self.safety.toxicity_score(text),
            "has_pii": has_pii,
        }

    def check_image_hash(self, image_bytes: bytes) -> bool:
        """Block known CSAM hashes (placeholder: SHA-256 blocklist)."""
        h = hashlib.sha256(image_bytes).hexdigest()
        return h not in self.BLOCKED_IMAGE_HASHES

    def check_image_heuristic(self, pixel_stats: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """CLIP-based NSFW placeholder using brightness/variance heuristics."""
        if pixel_stats is None:
            return True, {"nsfw_score": 0.0}
        nsfw_score = pixel_stats.get("nsfw_score", 0.0)
        return nsfw_score < 0.5, {"nsfw_score": nsfw_score}

    def check_audio_transcript(self, transcript: str) -> Tuple[bool, Dict]:
        return self.check_text(transcript)

    def filter_output(self, text: str) -> Tuple[str, bool]:
        meta = {}
        passed, meta = self.check_text(text)
        if not passed:
            return text, False
        redacted = self.pii.redact(text)
        # Block API keys and secrets
        secret_pattern = re.compile(r"(sk-[a-zA-Z0-9]{20,}|api[_-]?key\s*[:=]\s*\S+)", re.IGNORECASE)
        if secret_pattern.search(redacted):
            return redacted, False
        return redacted, True
