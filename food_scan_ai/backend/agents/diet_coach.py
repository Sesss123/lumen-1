# =====================================================================
# Tourist Culinary & Cultural Guide AI Agent (Savor Lanka / Trip Me)
# =====================================================================
# ශ්‍රී ලංකාවට එන විදේශීය සංචාරකයින්ට (Tourists) ආහාරයේ පෝෂණයට අමතරව,
# එහි සංස්කෘතික පසුබිම, සැර ප්‍රමාණය (Spice Level) සහ කෑමට ගන්නා නිවැරදි
# ආකාරය (Local Tips) පිළිබඳව උද්යෝගිමත් තොරතුරු ලබා දෙන AI ඒජන්තවරයා.
# =====================================================================

from typing import List, Dict, Any
from nutrition.db import NutritionFacts, DishComponent

from rag.food_knowledge_rag import FoodRAGSystem

# Initialize ChromaDB RAG Vector Store
food_rag_system = FoodRAGSystem(db_path="./chroma_db")

class TouristFoodGuideAgent:
    def check_spice_emergency(self, dish_name: str, components: List[Any]) -> Dict[str, Any]:
        """
        Lumen-1 RAG & Telemetry Inspired Spice Emergency Guide:
        අධික සැර කෑමක් දුටු විට RED Alert එකක් සහ සැර නිවාගන්නා ක්ෂණික විසඳුම් ලබා දෙයි.
        """
        combined_text = (dish_name + " " + " ".join([getattr(c, 'name', str(c)) for c in components])).lower()
        
        fiery_keywords = ["crab", "kottu", "devilled", "sambol", "black pork", "miris", "ambul thiyal", "spicy", "jaffna", "chilli", "curry", "hot"]
        
        is_spicy = any(kw in combined_text for kw in fiery_keywords)
        
        if is_spicy:
            return {
                "is_emergency": True,
                "alert_level": "RED",
                "badge_text": "🚨 SPICE ALERT: Fiery Sri Lankan Heat!",
                "remedy": "Do not drink ice water! Order King Coconut (Thambili), Curd & Treacle, or eat a spoonful of sugar/dairy immediately to neutralize the chili burn!",
                "telemetry_action": "TRIGGER_AR_WARNING_PULSE"
            }
        else:
            return {
                "is_emergency": False,
                "alert_level": "GREEN",
                "badge_text": "🟢 Mild & Safe",
                "remedy": "Safe and soothing for sensitive tourist stomachs!",
                "telemetry_action": "NORMAL_AR_DISPLAY"
            }

    def generate_coaching_advice(self, dish_name: str, nutrition: NutritionFacts, user_mode: str) -> str:
        # Retrieve rich cultural context from ChromaDB RAG Vector Store
        rag_context = food_rag_system.retrieve_knowledge(query=dish_name)

        # Customize based on tourist preference mode
        if user_mode == "spice_sensitive":
            return f"{rag_context}\n⚠️ Note for you: Please request 'Less Spicy / No Chili' from the chef if needed!"
        elif user_mode == "health_tourist":
            return f"{rag_context}\n🥗 Nutrition Info: Provides ~{nutrition.calories} kcal with {nutrition.protein_g}g clean protein."
        else:
            # Default rich tourist cultural response (from RAG)
            return f"{rag_context}"

    def get_fair_price_guide(self, dish_name: str) -> Dict[str, Any]:
        """
        Tourist Price & Scam Guard:
        කෑම Scan කළ සැනින් එහි සාමාන්‍ය දේශීය මිල ගණන් (LKR හා USD වලින්) පෙන්වා වංචා වලින් ආරක්ෂා කරයි.
        """
        clean_name = dish_name.lower()
        if any(w in clean_name for w in ["crab", "jaffna crab"]):
            lkr_min, lkr_max = 1800, 3500
        elif any(w in clean_name for w in ["kottu", "rice & curry", "lumprais", "biryani", "fried rice"]):
            lkr_min, lkr_max = 600, 1100
        elif any(w in clean_name for w in ["hopper", "egg hopper", "pittu", "string hopper"]):
            lkr_min, lkr_max = 150, 400
        elif any(w in clean_name for w in ["thambili", "king coconut"]):
            lkr_min, lkr_max = 150, 250
        elif any(w in clean_name for w in ["roti", "samosa", "cutlet", "wadai", "issawadai"]):
            lkr_min, lkr_max = 80, 200
        else:
            lkr_min, lkr_max = 400, 900

        usd_min = round(lkr_min / 300.0, 2)
        usd_max = round(lkr_max / 300.0, 2)

        return {
            "is_scam_guard_active": True,
            "fair_price_lkr": f"Rs. {lkr_min} - {lkr_max}",
            "fair_price_usd": f"~ ${usd_min} - ${usd_max} USD",
            "max_reasonable_lkr": lkr_max + 200,
            "advice": f"💡 Fair Local Price: Rs. {lkr_min} - {lkr_max} (~${usd_min}-${usd_max}). Avoid paying more than Rs. {lkr_max + 200}! Politely check standard local menu."
        }

    def generate_multilingual_voice(self, dish_name: str, lang: str = "en") -> Dict[str, Any]:
        """
        Multilingual Voice Tour Guide:
        ප්‍රංශ, ජර්මන්, රුසියානු හෝ චීන සංචාරකයින් සඳහා ඔවුන්ගේ මව් භාෂාවෙන් විස්තර සහ TTS Voice Tags ලබා දෙයි.
        """
        translations = {
            "fr": {
                "greeting": "Bon appétit!",
                "summary": f"{dish_name} est un plat traditionnel srilankais exquis, riche en épices des îles.",
                "tts_locale": "fr-FR"
            },
            "de": {
                "greeting": "Guten Appetit!",
                "summary": f"{dish_name} ist eine köstliche traditionelle Spezialität aus Sri Lanka mit exotischen Gewürzen.",
                "tts_locale": "de-DE"
            },
            "ru": {
                "greeting": "Приятного аппетита!",
                "summary": f"{dish_name} — это великолепное традиционное ланкийское блюдо с ароматными специями.",
                "tts_locale": "ru-RU"
            },
            "zh": {
                "greeting": "祝您用餐愉快！",
                "summary": f"{dish_name} 是一道美味的正宗斯里兰卡传统美食，融合了浓郁的当地香料。",
                "tts_locale": "zh-CN"
            },
            "en": {
                "greeting": "Ayubowan! Enjoy your meal!",
                "summary": f"{dish_name} is an authentic Sri Lankan delicacy bursting with exotic island spices.",
                "tts_locale": "en-US"
            }
        }
        selected = translations.get(lang.lower(), translations["en"])
        return {
            "language": lang.upper(),
            "greeting": selected["greeting"],
            "speech_text": f"{selected['greeting']} {selected['summary']}",
            "tts_locale": selected["tts_locale"],
            "auto_play_voice": True
        }

# Keep instance name 'diet_coach' for seamless backward compatibility with gateway.py
diet_coach = TouristFoodGuideAgent()
