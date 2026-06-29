# =====================================================================
# WebSocket Frame Processor (Throttling & Dedup)
# =====================================================================
# Mobile App එකේ කැමරාවෙන් එන Live Video Frames තත්පර 1.5 කට වරක්
# පමණක් සකසමින් Server එකේ හා Mobile Battery වේගය ආරක්ෂා කරයි.
# =====================================================================

import time
import base64
from typing import Optional, Dict, Any

class FrameProcessor:
    def __init__(self, throttle_ms: int = 1500):
        self.throttle_ms = throttle_ms
        self.last_processed_time: Dict[str, float] = {}
        self.last_frame_hash: Dict[str, str] = {}

    def should_process(self, client_id: str, image_base64: str) -> bool:
        """
        Throttling සහ Duplicate Frame පරීක්ෂාව (Frame Dedup).
        කැමරාව එකම තැන නොසෙල්වී තිබේ නම් නැවත AI එකට නොයවා ඉතිරි කරයි.
        """
        current_time = time.time() * 1000
        last_time = self.last_processed_time.get(client_id, 0)

        # 1. Throttling Check ( කාල සීමාව පරීක්ෂාව )
        if (current_time - last_time) < self.throttle_ms:
            return False

        # 2. Lightweight Hash Dedup Check
        frame_prefix = image_base64[:100] + image_base64[-100:] if len(image_base64) > 200 else image_base64
        if self.last_frame_hash.get(client_id) == frame_prefix:
            return False

        self.last_processed_time[client_id] = current_time
        self.last_frame_hash[client_id] = frame_prefix
        return True

    def decode_and_validate(self, image_base64: str) -> Optional[bytes]:
        """
        Base64 string එක නිවැරදිදැයි පරීක්ෂා කර bytes බවට හරවයි.
        """
        try:
            # Strip header if present e.g., data:image/jpeg;base64,...
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            return base64.b64decode(image_base64)
        except Exception:
            return None
