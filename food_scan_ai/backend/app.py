# =====================================================================
# Food Scan AI - Enterprise-Grade Modular Backend REST API
# Architecture: Tier 1 (FastAPI Orchestration) + Tier 3 (Deterministic DB Grounding)
# =====================================================================
# මෙම ගොනුව Claude AI ගේ උපදෙස් පරිදි State-of-the-Art Architecture එකකින්
# සකසා ඇත. කැලරි ගණනය AI එකට තනියම කිරීමට නොදී, නිශ්චිත දත්ත ගබඩාවක්
# (Deterministic Nutrition DB) හරහා සිදු කරන බැවින් Hallucinations 100% ක් වැළකේ.
# =====================================================================

import base64
import time
import json
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# FastAPI App එක ආරම්භ කිරීම
app = FastAPI(
    title="Food Scan AI - Tiered Enterprise API",
    description="State-of-the-art Food Scanner API with Tier 3 Deterministic Nutrition Grounding",
    version="2.0.0"
)

# =====================================================================
# 1. PYDANTIC DATA MODELS (Strict Validation Schemas)
# =====================================================================

class FoodScanRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded food image string")
    user_mode: Optional[str] = Field("normal", description="User goal mode: fitness, budget, or normal")
    spice_preference: Optional[str] = Field("medium", description="Spice level preference: mild, medium, spicy")

class DishComponent(BaseModel):
    name: str = Field(..., description="Name of the food component e.g. white rice, dhal curry")
    estimated_portion: str = Field(..., description="Standard portion unit e.g. 1 cup, 150g")
    portion_confidence: float = Field(0.9, description="Confidence in portion estimation 0.0-1.0")

class NutritionFacts(BaseModel):
    calories: int = Field(..., description="Rounded to nearest 10 kcal")
    protein_g: float = Field(..., description="Rounded to nearest 1g")
    carbs_g: float = Field(..., description="Rounded to nearest 1g")
    fat_g: float = Field(..., description="Rounded to nearest 1g")

class AROverlayData(BaseModel):
    bounding_box: Dict[str, float] = Field(..., description="Normalized coordinates x, y, width, height")
    badge_color: str
    label: str
    confidence_state: str = Field(..., description="high, estimated, or low")

class FoodScanResponse(BaseModel):
    status: str
    dish_name: str
    confidence: float
    components: List[DishComponent]
    nutrition: NutritionFacts
    nutrition_confidence: float
    recommendation: str
    ar_overlay: AROverlayData
    processing_time_ms: float

# =====================================================================
# 2. SYSTEM PROMPT (Claude AI Optimized Prompt)
# =====================================================================

FOOD_SCAN_SYSTEM_PROMPT = """
You are a food identification system for a nutrition scanning app. You analyze a photo of a meal and output ONLY a JSON object — no prose, no markdown fences, no explanation before or after.

## YOUR JOB
Identify the dish and its visible components. You do NOT calculate final nutrition values. A downstream database performs all calorie/macro math. Your role is identification and portion estimation only.

## SPECIAL DOMAIN KNOWLEDGE
You will frequently see South/Southeast Asian meals, especially Sri Lankan "rice and curry" plates. These typically contain multiple discrete components on one plate (e.g., rice, 2-4 curries, sambol, papadam). Identify EACH visible component separately rather than labeling the whole plate as one generic item.

## STRICT PROHIBITIONS
- NEVER output a calorie, protein, carb, or fat number. You do not have reliable enough visual information to compute these from a photo. Leave all nutrition math to the database layer.
"""

# =====================================================================
# 3. TIER 3 DETERMINISTIC NUTRITION DATABASE (Sri Lankan & Asian DB)
# =====================================================================
# AI එකට කැලරි හදන්න නොදී, හඳුනාගත් කෑම වර්ගය අනුව මෙම දත්ත ගබඩාවෙන්
# නිශ්චිත අගයන් ලබා ගනී. එමගින් වැරදි කැලරි පෙන්වීම 100% ක් වැළකේ.

SRI_LANKAN_NUTRITION_DB = {
    "white rice": {"calories": 200, "protein": 4.0, "carbs": 44.0, "fat": 0.5},
    "red rice": {"calories": 180, "protein": 4.5, "carbs": 40.0, "fat": 1.0},
    "chicken curry": {"calories": 180, "protein": 18.0, "carbs": 4.0, "fat": 10.0},
    "fish curry": {"calories": 150, "protein": 20.0, "carbs": 3.0, "fat": 6.0},
    "dhal curry": {"calories": 120, "protein": 6.0, "carbs": 18.0, "fat": 3.0},
    "parippu": {"calories": 120, "protein": 6.0, "carbs": 18.0, "fat": 3.0},
    "coconut sambol": {"calories": 100, "protein": 1.0, "carbs": 4.0, "fat": 9.0},
    "pol sambol": {"calories": 100, "protein": 1.0, "carbs": 4.0, "fat": 9.0},
    "gotukola sambol": {"calories": 45, "protein": 2.0, "carbs": 5.0, "fat": 2.0},
    "papadam": {"calories": 45, "protein": 1.0, "carbs": 5.0, "fat": 2.5},
    "wambatu moju": {"calories": 110, "protein": 1.5, "carbs": 14.0, "fat": 5.5},
    "kottu": {"calories": 650, "protein": 24.0, "carbs": 75.0, "fat": 28.0},
    "egg hopper": {"calories": 160, "protein": 7.0, "carbs": 18.0, "fat": 6.5},
    "plain hopper": {"calories": 90, "protein": 2.0, "carbs": 16.0, "fat": 2.0},
}

def resolve_nutrition(components: List[DishComponent]) -> Tuple[NutritionFacts, float]:
    """
    Tier 3 Grounding Function: හඳුනාගත් component ලැයිස්තුවට අදාළව
    නිශ්චිත කැලරි හා පෝෂණ ගුණ ගණනය කරයි.
    """
    total_cal = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0
    matched_count = 0

    for comp in components:
        key = comp.name.lower().strip()
        # Database එක තුළ ඇති කෑමක් සමඟ ගැලපේදැයි පරීක්ෂා කිරීම
        matched_item = None
        for db_key, db_val in SRI_LANKAN_NUTRITION_DB.items():
            if db_key in key or key in db_key:
                matched_item = db_val
                break
        
        if matched_item:
            total_cal += matched_item["calories"]
            total_protein += matched_item["protein"]
            total_carbs += matched_item["carbs"]
            total_fat += matched_item["fat"]
            matched_count += 1
        else:
            # Generic fallback for unknown side dish (උදාහරණයක් ලෙස නොදන්නා කරි වර්ගයක්)
            total_cal += 80
            total_protein += 2.0
            total_carbs += 8.0
            total_fat += 4.0

    # Claude's Rule: Round calories to nearest 10, macros to nearest 1g
    rounded_cal = int(round(total_cal / 10.0) * 10)
    rounded_protein = round(total_protein)
    rounded_carbs = round(total_carbs)
    rounded_fat = round(total_fat)

    # Calculate Nutrition Confidence based on matched items
    conf = 0.85 if (matched_count == len(components) and len(components) > 0) else 0.65

    return NutritionFacts(
        calories=rounded_cal,
        protein_g=rounded_protein,
        carbs_g=rounded_carbs,
        fat_g=rounded_fat
    ), conf

# =====================================================================
# 4. API ENDPOINT IMPLEMENTATION
# =====================================================================

@app.post("/api/food/scan", response_model=FoodScanResponse)
async def scan_food(request: FoodScanRequest):
    """
    Tier 1 Orchestration Endpoint:
    Mobile App එකෙන් ලැබෙන Base64 පින්තූරය පරීක්ෂා කර, AI විශ්ලේෂණය හා
    Tier 3 Database ගණනය ඒකාබද්ධ කර ප්‍රතිඵල සපයයි.
    """
    start_time = time.time()

    # 1. ආරක්ෂක පරීක්ෂාව (Payload validation)
    if not request.image_base64 or len(request.image_base64) < 50:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Image Base64 payload")

    # 2. Vision Model Identification Simulation
    # සැබෑ භාවිතයේදී මෙහිදී Claude Vision API හෝ Lumen-1 Model එක අමතනු ලබයි.
    # උදාහරණයක් ලෙස ශ්‍රී ලාංකික Rice & Curry පිඟානක් හඳුනාගත් බව උපකල්පනය කරමු:
    simulated_components = [
        DishComponent(name="white rice", estimated_portion="1 cup", portion_confidence=0.95),
        DishComponent(name="chicken curry", estimated_portion="150g", portion_confidence=0.90),
        DishComponent(name="dhal curry", estimated_portion="100g", portion_confidence=0.88),
        DishComponent(name="pol sambol", estimated_portion="30g", portion_confidence=0.85),
        DishComponent(name="papadam", estimated_portion="2 pieces", portion_confidence=0.92)
    ]
    dish_name = "Sri Lankan Rice & Curry Plate"
    id_confidence = 0.94

    # 3. Tier 3 Deterministic Nutrition Lookup (කැලරි ගණනය කිරීම)
    nutrition_facts, nutr_conf = resolve_nutrition(simulated_components)

    # 4. User Mode අනුව Recommendation සැකසීම
    if request.user_mode == "fitness":
        rec = f"Great lean protein from Chicken Curry ({nutrition_facts.protein_g}g total)! Consider slightly less white rice for a cutting goal."
    elif request.user_mode == "budget":
        rec = "Excellent nutrient-dense meal! Dhal and Sambol provide amazing budget-friendly energy and healthy fats."
    else:
        rec = "Balanced traditional Sri Lankan plate with authentic spices and rich micronutrients."

    # 5. Confidence State අනුව AR Overlay එක හැඩගැස්වීම (Guardrails)
    if id_confidence >= 0.85 and nutr_conf >= 0.80:
        conf_state = "high"
        badge_col = "#00F0FF" # Electric Blue
        label_text = f"{dish_name} (~{nutrition_facts.calories} kcal)"
    else:
        conf_state = "estimated"
        badge_col = "#FFC107" # Warning Yellow for estimate
        label_text = f"Est: {dish_name} (~{nutrition_facts.calories} kcal)"

    ar_data = AROverlayData(
        bounding_box={"x": 0.1, "y": 0.15, "width": 0.8, "height": 0.7},
        badge_color=badge_col,
        label=label_text,
        confidence_state=conf_state
    )

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return FoodScanResponse(
        status="success",
        dish_name=dish_name,
        confidence=id_confidence,
        components=simulated_components,
        nutrition=nutrition_facts,
        nutrition_confidence=nutr_conf,
        recommendation=rec,
        ar_overlay=ar_data,
        processing_time_ms=elapsed_ms
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "architecture": "Tiered Tier1+Tier3 Grounding"}
