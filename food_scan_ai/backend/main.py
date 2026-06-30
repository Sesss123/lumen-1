# =====================================================================
# Food Scan AI - Modular Enterprise Backend Entry Point
# =====================================================================
# මෙම ප්‍රධාන entry point එක හරහා HTTP REST APIs සහ Real-Time WebSocket
# streaming endpoints සියල්ල ක්‍රියාත්මක වේ.
# =====================================================================

import time
from fastapi import FastAPI, WebSocket, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
from config import settings
from ws.gateway import websocket_endpoint
from vision.multimodal import analyze_food_image
from nutrition.db import resolve_nutrition, DishComponent, NutritionFacts
from agents.diet_coach import diet_coach

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="State-of-the-Art Modular Real-Time Food Scanner Backend"
)

# Request & Response Pydantic Models for HTTP Fallback Endpoint
class HTTPScanRequest(BaseModel):
    image_base64: str
    user_mode: Optional[str] = "normal"
    spice_preference: Optional[str] = "medium"

# 1. Real-Time WebSocket Streaming Endpoint for Mobile Apps
@app.websocket("/api/food/scan/live")
@app.websocket("/ws/scan")
async def live_scan_websocket(websocket: WebSocket):
    await websocket_endpoint(websocket)

# 2. Standard HTTP POST Endpoint (For non-streaming / fallback calls)
@app.post("/api/food/scan")
async def scan_food_http(request: HTTPScanRequest):
    start_t = time.time() * 1000
    if not request.image_base64 or len(request.image_base64) < 50:
        raise HTTPException(status_code=400, detail="Invalid Base64 payload")

    dish_name, conf, components = await analyze_food_image(request.image_base64)
    nutr_facts, nutr_conf = resolve_nutrition(components)
    advice = diet_coach.generate_coaching_advice(dish_name, nutr_facts, request.user_mode)
    
    badge_col = settings.AR_COLOR_HIGH_CONF if conf >= 0.85 else settings.AR_COLOR_ESTIMATED
    elapsed_ms = round((time.time() * 1000) - start_t, 2)

    return {
        "status": "success",
        "dish_name": dish_name,
        "confidence": conf,
        "components": [c.dict() for c in components],
        "nutrition": nutr_facts.dict(),
        "nutrition_confidence": nutr_conf,
        "recommendation": advice,
        "ar_overlay": {
            "bounding_box": {"x": 0.1, "y": 0.15, "width": 0.8, "height": 0.7},
            "badge_color": badge_col,
            "label": f"{dish_name} (~{nutr_facts.calories} kcal)",
            "confidence_state": "high" if conf >= 0.85 else "estimated"
        },
        "processing_time_ms": elapsed_ms
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "architecture": "Modular Enterprise (ws/ + agents/ + rag/ + vision/ + nutrition/)"
    }
