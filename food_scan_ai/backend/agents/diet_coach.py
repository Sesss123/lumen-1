# =====================================================================
# Multi-turn Diet Coach AI Agent
# =====================================================================
# පරිශීලකයාගේ ප්‍රශ්න වලට සහ හඳුනාගත් ආහාර වලට සෞඛ්‍ය සම්පන්න උපදෙස්
# ලබා දෙන බුද්ධිමත් ඒජන්තවරයා.
# =====================================================================

from typing import List, Dict, Any
from nutrition.db import NutritionFacts, DishComponent
from agents.tools import search_healthy_alternatives

class DietCoachAgent:
    def generate_coaching_advice(self, dish_name: str, nutrition: NutritionFacts, user_mode: str) -> str:
        alts = search_healthy_alternatives(dish_name, user_mode)
        alt_title = alts[0]["title"] if alts else "Red Rice & Baked Fish"

        if user_mode == "fitness":
            return f"Great lean protein source ({nutrition.protein_g}g)! To optimize cutting, consider substituting part of the carbs with {alt_title}."
        elif user_mode == "budget":
            return f"Excellent energy density! This meal gives you sustainable calories. Next time, try adding Gotukola Sambol for affordable micronutrients."
        else:
            return f"Balanced traditional meal with authentic spices! Contains around {nutrition.calories} kcal."

diet_coach = DietCoachAgent()
