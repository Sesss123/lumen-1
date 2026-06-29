# =====================================================================
# Food Scan AI - Enterprise Configuration Module
# =====================================================================
# මෙම ගොනුව මගින් මුළු පද්ධතියේම සැකසුම් (Settings), API keys, සහ 
# Mobile App එකේ වේගය පාලනය කරන Throttling සීමාවන් කළමනාකරණය කරයි.
# =====================================================================

import os
from pydantic import BaseModel

class AppConfig(BaseModel):
    APP_NAME: str = "Food Scan AI Enterprise Backend"
    VERSION: str = "3.0.0"
    PORT: int = 8000
    HOST: str = "0.0.0.0" # Mobile App එකට Connect වීමට 0.0.0.0 ලබා දී ඇත
    
    # WebSocket & Real-Time Mobile Throttling Settings
    FRAME_THROTTLE_MS: int = 1500  # තත්පර 1.5 කට වරක් පමණක් Frame එකක් සකසයි (Speed Optimization)
    MAX_FRAME_SIZE_KB: int = 500   # උපරිම පින්තූර ප්‍රමාණය KB වලින්
    
    # AI & RAG Settings
    ENABLE_REDIS_CACHE: bool = False # Local testing සඳහා False කර ඇත
    ENABLE_VECTOR_STORE: bool = False # Local Fallback mode
    
    # Electric Blue UI/UX Theme Colors for AR Overlays
    AR_COLOR_HIGH_CONF: str = "#00F0FF" # Electric Blue
    AR_COLOR_ESTIMATED: str = "#FFC107" # Warning Yellow
    AR_COLOR_LOW_CONF: str = "#FF3366"  # Alert Red

settings = AppConfig()
