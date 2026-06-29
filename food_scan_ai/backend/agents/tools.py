# =====================================================================
# Agent Tools (Nutrition Lookup & RAG Search)
# =====================================================================
# Diet Coach AI Agent එකට භාවිතා කිරීමට ඇති මෙවලම්.
# =====================================================================

from typing import Dict, Any, List
from nutrition.db import resolve_nutrition, DishComponent
from rag.recipe_retriever import recipe_retriever

def lookup_dish_nutrition(components: List[DishComponent]) -> Dict[str, Any]:
    facts, conf = resolve_nutrition(components)
    return {
        "calories": facts.calories,
        "protein_g": facts.protein_g,
        "carbs_g": facts.carbs_g,
        "fat_g": facts.fat_g,
        "confidence": conf
    }

def search_healthy_alternatives(dish_name: str, user_mode: str) -> List[Dict[str, Any]]:
    return recipe_retriever.retrieve_alternatives(dish_name, user_mode)
