# Food Scan AI - Modular Sub-project

මෙම Sub-project එක ප්‍රධාන Lumen-1 පද්ධතියෙන් වෙන්ව (Isolated) ක්‍රියාත්මක වන අතර, ඔබගේ Mobile App එක වෙත උපරිම වේගය (Speed) හා AI විශ්ලේෂණය ලබා දීමට නිර්මාණය කර ඇත.

## 📁 ගොනු ව්‍යුහය (Directory Structure)
```
food_scan_ai/
└── backend/
    ├── app.py                  # FastAPI Backend API Endpoint (POST /api/food/scan)
    ├── requirements.txt        # Python Dependencies (FastAPI, Uvicorn, etc.)
    └── test_scan_api.py        # API Test Script (API එක නිවැරදිදැයි පරීක්ෂා කිරීමට)
```

---

## 🚀 Terminal Commands (User Instructions)
**අවවාදයයි:** AI Agent කිසිදු Terminal Command එකක් Run නොකරයි. කරුණාකර පහත Commands ඔබගේ Terminal එකේ Run කරන්න.

### 1. Backend Server එක Run කිරීම (FastAPI)
ඔබගේ Terminal එක (PowerShell හෝ CMD) විවෘත කර පහත විධානය ක්‍රියාත්මක කරන්න:

```powershell
# 1. Backend folder එකට යන්න
cd "c:\Users\sehas\Downloads\ai model\food_scan_ai\backend"

# 2. Uvicorn server එක run කරන්න
python -m uvicorn app:app --reload --port 8000
```
Server එක සාර්ථකව ක්‍රියාත්මක වූ පසු, Web Browser එකෙන් `http://127.0.0.1:8000/docs` වෙත ගොස් API එක Test කළ හැක.

---

### 2. ඔබගේ Mobile App එක සම්බන්ධ කිරීම (API Endpoint)
ඔබගේ App එකෙන් පහත Endpoint එක වෙත POST Request එකක් යොමු කරන්න:
- **URL:** `http://<Your-Server-IP>:8000/api/food/scan`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Payload:**
```json
{
  "image_base64": "iVBORw0KGgo...",
  "user_mode": "fitness",
  "spice_preference": "spicy"
}
```
