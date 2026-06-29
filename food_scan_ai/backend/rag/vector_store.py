# =====================================================================
# Vector Store Client (Qdrant / Local Memory Fallback)
# =====================================================================
# සෞඛ්‍ය සම්පන්න ආහාර සහ වට්ටෝරු (Healthy Recipes) ගබඩා කර වේගයෙන්
# සෙවීම සඳහා භාවිතා කරයි.
# =====================================================================

from typing import List, Dict, Any

class VectorStore:
    def __init__(self):
        self.recipes = [
            {"title": "Red Rice & Baked Chicken", "desc": "High protein cutting substitute for white rice & fried curry.", "tags": ["fitness", "chicken"]},
            {"title": "Gotukola & Dhal Bowl", "desc": "Ultra nutrient dense budget meal rich in iron and fiber.", "tags": ["budget", "dhal"]},
            {"title": "Steamed String Hoppers with Fish Ambul Thiyal", "desc": "Authentic low-fat Sri Lankan dinner.", "tags": ["normal", "fish"]}
        ]

    def search_similar(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for r in self.recipes:
            if any(t in query_lower for t in r["tags"]) or any(w in r["title"].lower() for w in query_lower.split()):
                results.append(r)
        return results[:limit] if results else self.recipes[:1]

vector_store = VectorStore()
