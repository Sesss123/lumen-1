"""Data filtering: dedup, quality, safety, PII."""

import hashlib
import re
from typing import Iterable, List, Optional, Set

# MinHash-style deduplication via SHA-256 n-gram fingerprints
NGRAM_SIZE = 5


class DedupFilter:
    """MinHash-inspired deduplication using n-gram Jaccard approximation."""

    def __init__(self, jaccard_threshold: float = 0.8):
        self.jaccard_threshold = jaccard_threshold
        self._seen_hashes: Set[str] = set()
        self._ngram_sets: List[Set[str]] = []

    def _ngrams(self, text: str) -> Set[str]:
        words = text.lower().split()
        if len(words) < NGRAM_SIZE:
            return {hashlib.sha256(text.encode()).hexdigest()}
        return {
            hashlib.sha256(" ".join(words[i : i + NGRAM_SIZE]).encode()).hexdigest()
            for i in range(len(words) - NGRAM_SIZE + 1)
        }

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def is_duplicate(self, text: str) -> bool:
        ngrams = self._ngrams(text)
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        if content_hash in self._seen_hashes:
            return True

        for seen in self._ngram_sets:
            if self._jaccard(ngrams, seen) >= self.jaccard_threshold:
                return True

        self._seen_hashes.add(content_hash)
        self._ngram_sets.append(ngrams)
        return False

    def filter_batch(self, texts: Iterable[str]) -> List[str]:
        return [t for t in texts if not self.is_duplicate(t)]


class QualityFilter:
    """Heuristic quality scoring (perplexity proxy + length gates)."""

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 100_000,
        min_alpha_ratio: float = 0.7,
        max_repetition_ratio: float = 0.3,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_alpha_ratio = min_alpha_ratio
        self.max_repetition_ratio = max_repetition_ratio

    def score(self, text: str) -> float:
        if len(text) < self.min_length or len(text) > self.max_length:
            return 0.0
        alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha < self.min_alpha_ratio:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < (1 - self.max_repetition_ratio):
            return 0.0
        return min(1.0, unique_ratio * alpha)

    def passes(self, text: str, threshold: float = 0.5) -> bool:
        return self.score(text) >= threshold


class SafetyFilter:
    """NSFW and toxicity keyword filter (production would use classifiers)."""

    BLOCKED_PATTERNS = [
        r"\b(child\s*(porn|abuse|sexual))\b",
        r"\b(terrorist\s*attack\s*how\s*to)\b",
        r"\b(make\s*a\s*bomb)\b",
    ]

    def __init__(self, toxicity_threshold: float = 0.3):
        self.toxicity_threshold = toxicity_threshold
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.BLOCKED_PATTERNS]

    def toxicity_score(self, text: str) -> float:
        hits = sum(1 for p in self._compiled if p.search(text))
        return min(1.0, hits * 0.5)

    def passes(self, text: str) -> bool:
        return self.toxicity_score(text) < self.toxicity_threshold


class PIIFilter:
    """PII redaction for emails, phones, SSNs."""

    PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    }

    def redact(self, text: str) -> str:
        for name, pattern in self.PATTERNS.items():
            text = pattern.sub(f"[REDACTED_{name.upper()}]", text)
        return text

    def contains_pii(self, text: str) -> bool:
        return any(p.search(text) for p in self.PATTERNS.values())
