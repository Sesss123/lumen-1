# =====================================================================
# WebSocket Gateway & Frame Router
# =====================================================================
# Mobile App එකෙන් එන Live Frames වේගයෙන් විශ්ලේෂණය කර, AR Overlays
# සහ Badges මිලි තත්පර ගණනකින් නැවත Mobile App එක වෙත යොමු කරයි.
# =====================================================================

import json
import time
from fastapi import WebSocket, WebSocketDisconnect
from ws.frame_processor import FrameProcessor
from vision.multimodal import analyze_food_image
from nutrition.db import resolve_nutrition
from agents.diet_coach import diet_coach
from config import settings

frame_processor = FrameProcessor(throttle_ms=settings.FRAME_THROTTLE_MS)

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(id(websocket))
    print(f"📱 Mobile Client Connected to Real-Time Scanner: {client_id}")

    try:
        while True:
            data = await websocket.receive_text()
            start_t = time.time() * 1000

            try:
                payload = json.loads(data)
                image_base64 = payload.get("image_base64", "")
                user_mode = payload.get("user_mode", "normal")
            except Exception:
                await websocket.send_json({"status": "error", "message": "Invalid JSON payload"})
                continue

            # Throttling & Duplicate frame check
            if not frame_processor.should_process(client_id, image_base64):
                await websocket.send_json({"status": "skipped", "message": "Frame throttled or identical"})
                continue

            # 1. Vision Analysis
            dish_name, conf, components = await analyze_food_image(image_base64)

            # 2. Tier 3 Deterministic Nutrition Lookup
            nutr_facts, nutr_conf = resolve_nutrition(components)

            # 3. AI Coaching Advice
            advice = diet_coach.generate_coaching_advice(dish_name, nutr_facts, user_mode)

            # 4. AR Overlay Formatting (Electric Blue Colors)
            badge_col = settings.AR_COLOR_HIGH_CONF if conf >= 0.85 else settings.AR_COLOR_ESTIMATED
            
            elapsed_ms = round((time.time() * 1000) - start_t, 2)

            response_payload = {
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

            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        print(f"📴 Mobile Client Disconnected: {client_id}")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
