# =====================================================================
# Vision & Embedding Cache Module
# =====================================================================
# පෙර විශ්ලේෂණය කළ කෑම වර්ග මතක තබා ගෙන (Speculative Prefetch & Cache)
# Mobile App එකට මිලි තත්පර ගණනකින් ප්‍රතිචාර ලබා දෙයි.
# =====================================================================

from typing import Optional, Dict, Any

class VisionCache:
    def __init__(self):
        # In-memory LRU cache fallback when Redis is not active locally
        self._memory_cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._memory_cache.get(key)

    def set(self, key: str, value: Any):
        if len(self._memory_cache) > 500:
            # Clear oldest items if memory cache gets large
            self._memory_cache.clear()
        self._memory_cache[key] = value

vision_cache = VisionCache()
