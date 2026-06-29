# =====================================================================
# Food Scan AI - Upgraded Tiered API Test Client
# =====================================================================
# මෙම ස්ක්‍රිප්ට් එක මගින් Tier 3 Deterministic Nutrition Lookup සහිත
# නවතම Backend API එක අත්හදා බැලිය හැක.
# =====================================================================

import urllib.request
import json
import base64

API_URL = "http://127.0.0.1:8000/api/food/scan"

# අත්හදා බැලීම සඳහා ව්‍යාජ (Dummy) Base64 Image දත්තයක්
dummy_image_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==" * 50
base64_str = base64.b64encode(dummy_image_data).decode('utf-8')

payload = {
    "image_base64": base64_str,
    "user_mode": "fitness",
    "spice_preference": "spicy"
}

print("🍽️ Sending Request to Upgraded Tiered Food Scan API...")

try:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        print("\n✅ Enterprise Tiered API Response Received!\n")
        print(f"🍛 Dish Name         : {result['dish_name']}")
        print(f"🎯 ID Confidence     : {int(result['confidence'] * 100)}%")
        print(f"🛡️ Nutrition Conf    : {int(result['nutrition_confidence'] * 100)}% (Deterministic DB Lookup)")
        print(f"🔥 Calories          : {result['nutrition']['calories']} kcal (Rounded to nearest 10)")
        print(f"💪 Macros            : {result['nutrition']['protein_g']}g P | {result['nutrition']['carbs_g']}g C | {result['nutrition']['fat_g']}g F")
        print("\n🥗 Identified Components (Tier 3 Breakdown):")
        for comp in result['components']:
            print(f"   ▫️ {comp['name'].title()} ({comp['estimated_portion']}) - {int(comp['portion_confidence']*100)}% conf")
        print(f"\n💡 Recommendation    : {result['recommendation']}")
        print(f"🔵 AR Overlay Badge  : {result['ar_overlay']['label']} [{result['ar_overlay']['confidence_state'].upper()}]")
        print(f"⏱️ Processing Time   : {result['processing_time_ms']} ms\n")
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print(" කරුණාකර 'python -m uvicorn main:app --reload --port 8000' මගින් Server එක Run වී ඇත්දැයි පරීක්ෂා කරන්න.")
