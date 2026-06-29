# =====================================================================
# Text & Vision Embedding Module (RAG)
# =====================================================================
# කෑම වර්ග සහ ප්‍රශ්න Vector Embeddings බවට පත් කරයි.
# =====================================================================

from typing import List

class Embedder:
    def embed_text(self, text: str) -> List[float]:
        """
        Lightweight dummy vector embedding representation for fallback testing.
        """
        # Return 128-dim normalized dummy vector based on string hash
        h = hash(text)
        return [float((h >> (i % 16)) & 1) for i in range(128)]

embedder = Embedder()
