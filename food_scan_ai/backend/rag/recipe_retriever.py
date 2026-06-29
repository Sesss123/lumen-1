# =====================================================================
# Recipe Retriever & Re-ranker
# =====================================================================
# හඳුනාගත් පිඟානේ ඇති කෑම වලට ගැළපෙන වඩාත් සෞඛ්‍ය සම්පන්න
# විකල්ප (Healthy Alternatives) තෝරා දෙයි.
# =====================================================================

from typing import List, Dict, Any
from rag.vector_store import vector_store

class RecipeRetriever:
    def retrieve_alternatives(self, dish_name: str, user_mode: str) -> List[Dict[str, Any]]:
        query = f"{dish_name} {user_mode}"
        return vector_store.search_similar(query)

recipe_retriever = RecipeRetriever()
