"""
generate_sft_from_json.py
=========================
Existing tripme JSON files valata irun disambiguation-safe SFT training data generate karanawa.

Usage in Google Colab:
    1. Upload this script to Colab
    2. Mount Google Drive (or upload JSON files directly)
    3. exec(open("generate_sft_from_json.py").read())

Output files:
    - tripme_merged_all.json          → All locations merged & deduplicated
    - tripme_sft_training.jsonl       → SFT Alpaca-format Q&A pairs (Sinhala + English)
    - tripme_sft_training_si.jsonl    → Sinhala-only Q&A pairs
    - tripme_sft_training_en.jsonl    → English-only Q&A pairs
"""

import os
import sys
import json
import hashlib

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: Install dependencies if in Colab
# ─────────────────────────────────────────────────────────────────────────────
# Kaggle environment check first (most reliable)
if os.path.exists("/kaggle/input"):
    IN_KAGGLE = True
    IN_COLAB  = False
    print("🏆 Kaggle environment detected.")
else:
    try:
        import google.colab
        IN_COLAB  = True
        IN_KAGGLE = False
        print("🌐 Google Colab detected.")
    except ImportError:
        IN_COLAB = False
        IN_KAGGLE = False
        print("💻 Local environment detected.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — JSON files ekata path list karanna
# ─────────────────────────────────────────────────────────────────────────────

# Auto-select base path depending on environment
if IN_KAGGLE:
    # Exact path found from Kaggle environment
    GDRIVE_BASE = "/kaggle/input/datasets/sehaspathirana/tripmedata"
elif IN_COLAB:
    # Google Drive: My Drive > Places > json
    GDRIVE_BASE = "/content/drive/MyDrive/Places/json"
else:
    # Local PC
    GDRIVE_BASE = "."   # current folder

# JSON file names (all in same folder)
JSON_FILES = [
    "tripme_database_complete_water-falls.json",
    "tripme_database_complete_viwe-point.json",
    "tripme_database_complete_tea estate.json",
    "tripme_database_complete_Religious Places.json",
    "tripme_database_complete_beach.json",
    "tripme_database_complete_Adventure Park.json",
    # Meka mata add karagann puluvann — new files:
    # "tripme_database_complete_NEW.json",
]

# Output paths
OUTPUT_MERGED_JSON   = "tripme_merged_all.json"
OUTPUT_SFT_ALL       = "tripme_sft_training.jsonl"        # Sinhala + English mixed
OUTPUT_SFT_SINHALA   = "tripme_sft_training_si.jsonl"     # Sinhala only
OUTPUT_SFT_ENGLISH   = "tripme_sft_training_en.jsonl"     # English only

# ─────────────────────────────────────────────────────────────────────────────
# DISAMBIGUATION KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

# Multi-region entities: Rivers, roads that span multiple districts.
MULTI_REGION_ENTITIES = {
    "mahaweli":   ["Nuwara Eliya", "Kandy", "Matale", "Polonnaruwa", "Trincomalee"],
    "kelani":     ["Nuwara Eliya", "Kegalle", "Colombo"],
    "kalu":       ["Ratnapura", "Kalutara"],
    "gin":        ["Ratnapura", "Galle"],
    "walawe":     ["Ratnapura", "Hambantota"],
    "deduru":     ["Kurunegala", "Puttalam"],
    "a9 highway": ["Colombo", "Gampaha", "Kurunegala", "Anuradhapura", "Vavuniya", "Kilinochchi", "Jaffna"],
    "a1 highway": ["Colombo", "Kalutara", "Galle", "Matara", "Hambantota"],
}

# Known confusing/ambiguous place pairs with clear notes.
AMBIGUOUS_PLACE_NOTES = {
    "kandy lake":       "Kandy Lake (නුවර වැව) is in Kandy city, Kandy District. NOT in Nuwara Eliya.",
    "නුවර වැව":        "නුවර වැව = Kandy Lake. මහනුවර (Kandy) දිස්ත්‍රික්කයේ ය. නුවරඑළිය නොවේ.",
    "galle face":       "Galle Face Green is in Colombo. Not in Galle city.",
    "victoria":         "Victoria Reservoir = Kandy District. Victoria Park = Nuwara Eliya. Different places.",
    "gregory":          "Lake Gregory = Nuwara Eliya. Not the same as Kandy Lake (නුවර වැව).",
    "sigiriya":         "Sigiriya is in Matale District, not Polonnaruwa District.",
    "pidurutalagala":   "Pidurutalagala (highest peak) is in Nuwara Eliya District.",
    "adam":             "Adam's Peak (ශ්‍රී පාද) is in Ratnapura District, not Nuwara Eliya.",
    "රාවණා":           "රාවණා ඇල්ල (Ravana Falls) Ella, Badulla District ලේ ඇත. Ravana is also a historical figure — different context.",
    "ella":             "Ella town is in Badulla District (Uva Province). Not in Nuwara Eliya Province.",
    "anuradhapura":     "Anuradhapura is both a district AND a city. Ancient lakes (Tissa Wewa, Nuwara Wewa) are inside the city.",
    "trincomalee":      "Trincomalee is in Eastern Province. Not to be confused with Trincomalee road in Kandy.",
    "හෝර්ටන්":         "Horton Plains is in Nuwara Eliya District. Not related to Horton Place, Colombo.",
    "horton":           "Horton Plains National Park is in Nuwara Eliya District. Horton Place is a road in Colombo 7.",
}

# Historical figure disambiguation — for Q&A covering rulers/persons with same name.
HISTORICAL_FIGURES = {
    "parakramabahu": {
        "note": "ශ්‍රී ලංකා ඉතිහාසයේ 'පරාක්‍රමබාහු' නමින් රජවරු 12ක් ඉදලා ඇත. / Sri Lanka had 12 kings named Parakramabahu.",
        "notable": {
            "I":  "පරාක්‍රමසමුද්‍රය ඉදිකළේ පළමු පරාක්‍රමබාහු (ක්‍රි.ව. 1153–1186). / Parakramabahu I (1153–1186) built Parakrama Samudraya, Polonnaruwa.",
            "II": "දෙවන පරාක්‍රමබාහු (ක්‍රි.ව. 1236–1270) — Dambadeniya රාජධානිය.",
        }
    },
    "vijayabahu": {
        "note": "ශ්‍රී ලංකා ඉතිහාසයේ 'විජයබාහු' නමින් රජවරු කිහිප දෙනෙකු ඉදලා ඇත. / Multiple kings named Vijayabahu in Sri Lankan history.",
        "notable": {
            "I": "පළමු විජයබාහු (ක්‍රි.ව. 1055–1110) — Chola occupation ට එරෙහිව ජයග්‍රහණය.",
        }
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect_multi_region(name: str) -> list:
    """Returns list of districts if place spans multiple regions, else []."""
    name_lower = name.lower()
    for keyword, districts in MULTI_REGION_ENTITIES.items():
        if keyword in name_lower:
            return districts
    return []

def get_disambiguation_suffix(name: str) -> str:
    """Returns a disambiguation note string if place is in AMBIGUOUS_PLACE_NOTES."""
    name_lower = name.lower()
    for key, note in AMBIGUOUS_PLACE_NOTES.items():
        if key.lower() in name_lower:
            return f" ⚠️ Note: {note}"
    return ""

def get_historical_note(name: str) -> str:
    """Returns historical figure disambiguation note if applicable."""
    name_lower = name.lower()
    for fig_key, fig_data in HISTORICAL_FIGURES.items():
        if fig_key in name_lower:
            return fig_data["note"]
    return ""

def parse_and_estimate_budget(record: dict) -> dict:
    """
    Parses entry ticket price and estimates food, transport, and overall budget categories in LKR.
    """
    ticket_str = str(record.get("ticket_price", record.get("ticket", "Free"))).strip()
    popularity = str(record.get("tourist_popularity", "Medium")).strip()
    road = str(record.get("road_condition", "Paved")).strip().lower()

    # 1. Entry fee parsing / clean up
    entry_fee_si = ""
    entry_fee_en = ""
    is_free = False
    
    ticket_lower = ticket_str.lower()
    if "free" in ticket_lower or "no cost" in ticket_lower or ticket_str == "0" or ticket_str == "":
        entry_fee_si = "නොමිලේ (Free)"
        entry_fee_en = "Free of charge"
        is_free = True
        entry_lkr = 0
    else:
        entry_fee_si = f"ප්‍රවේශ ගාස්තුව {ticket_str} කි."
        entry_fee_en = f"Entry fee is {ticket_str}."
        import re
        nums = re.findall(r'\d+', ticket_str.replace(",", ""))
        if nums:
            entry_lkr = int(nums[0])
        else:
            entry_lkr = 500  # Default estimate if not parseable
            
    # 2. Food cost nearby estimation (in LKR per meal per person)
    if popularity == "High":
        food_si = "ප්‍රදේශයේ ආහාර මිල තරමක් වැඩි විය හැක (එක් අයෙකුට වේලක් සඳහා රු. 800 - 1500 පමණ)"
        food_en = "Food costs nearby can be slightly high (approx. LKR 800 - 1500 per meal per person)"
        food_lkr = 1100
    else:
        food_si = "සාමාන්‍ය දේශීය අවන්හල් වලින් ආහාර ලබා ගත හැක (එක් අයෙකුට වේලක් සඳහා රු. 400 - 800 පමණ)"
        food_en = "Standard local eateries are available (approx. LKR 400 - 800 per meal per person)"
        food_lkr = 600

    # 3. Transport cost nearby estimation (Tuk-tuk/Bus from closest town/hub)
    if "off-road" in road or "rough" in road or "difficult" in road:
        transport_si = "මාර්ගය අපහසු බැවින් ත්‍රීවීල් (Tuk-tuk) හෝ 4WD රථයක් සඳහා රු. 1500 - 3000 අතර මුදලක් වැය විය හැක."
        transport_en = "Due to rough road conditions, a Tuk-tuk or 4WD hire from the nearest town may cost LKR 1500 - 3000."
        trans_lkr = 2200
    else:
        transport_si = "ප්‍රධාන මාර්ගයේ සිට බස් රථයකින් හෝ ත්‍රීවීල් (Tuk-tuk) රථයකින් රු. 300 - 800 අතර මුදලකට ළඟා විය හැක."
        transport_en = "Easily accessible via bus or a short Tuk-tuk ride from the nearest town (approx. LKR 300 - 800)."
        trans_lkr = 500

    # 4. Total Budget Level
    total_est = entry_lkr + food_lkr + trans_lkr
    if total_est < 1500:
        level_si = "අඩු වියදම් (Budget-friendly)"
        level_en = "Budget-friendly / Low cost"
    elif total_est < 5000:
        level_si = "මධ්‍යම වියදම් (Mid-range)"
        level_en = "Mid-range"
    else:
        level_si = "ඉහළ වියදම් (High budget / Premium)"
        level_en = "High budget / Expensive"

    return {
        "is_free": is_free,
        "entry_fee_si": entry_fee_si,
        "entry_fee_en": entry_fee_en,
        "food_si": food_si,
        "food_en": food_en,
        "transport_si": transport_si,
        "transport_en": transport_en,
        "level_si": level_si,
        "level_en": level_en,
        "total_est_lkr": total_est
    }


def generate_qa_pairs(record: dict) -> tuple:
    """
    Eka location record ekata SFT Q&A pairs (Sinhala, English, and Singlish/Colloquial) generate karanawa.
    Returns (sinhala_pairs, english_pairs, colloquial_pairs) as lists of dicts.
    Each dict: {"instruction": str, "input": str, "output": str}
    """
    name       = str(record.get("name", "Unknown")).strip()
    district   = str(record.get("district_id", record.get("district", "Unknown"))).strip()
    province   = str(record.get("province_id", record.get("province", "Unknown"))).strip()
    category   = str(record.get("category_id", record.get("category", "Place"))).strip()
    lat        = float(record.get("lat", record.get("latitude", 0.0)) or 0.0)
    lng        = float(record.get("lng", record.get("longitude", 0.0)) or 0.0)
    desc       = str(record.get("description", "")).strip()
    best_time  = str(record.get("best_time_to_visit", "All year round")).strip()
    activities = str(record.get("activities", "Sightseeing")).strip()
    popularity = str(record.get("tourist_popularity", "Medium")).strip()
    ticket     = str(record.get("ticket_price", "Free")).strip()
    safety     = str(record.get("safety_level", "Safe")).strip()
    road       = str(record.get("road_condition", "Paved")).strip()

    disambig  = get_disambiguation_suffix(name)
    hist_note = get_historical_note(name)
    spans     = detect_multi_region(name)
    
    # Parse budget
    budget = parse_and_estimate_budget(record)

    si_pairs = []
    en_pairs = []
    col_pairs = []  # Singlish & colloquial/casual pairs

    pop_map = {"High": "ඉතා ජනප්‍රිය", "Medium": "මධ්‍යම ජනප්‍රිය", "Low": "අඩු ජනප්‍රිය"}
    pop_si  = pop_map.get(popularity, popularity)
    
    family = str(record.get("family_friendly", "yes")).lower()
    is_family = "yes" in family
    
    monsoon = str(record.get("monsoon_note", "None")).strip()
    rain    = str(record.get("rain_sensitivity", "Safe")).strip()

    # ── 1. Location & Access Q&A ─────────────────────────────────────────────
    si_pairs.append({
        "instruction": f"{name} කොහෙද?",
        "input": "",
        "output": f"{name} ශ්‍රී ලංකාවේ {province} පළාතේ, {district} දිස්ත්‍රික්කයේ පිහිටා ඇත. (GPS: {lat:.4f}°N, {lng:.4f}°E){disambig}"
    })
    en_pairs.append({
        "instruction": f"Where is {name} located?",
        "input": "",
        "output": f"{name} is located in {district} District, {province} Province, Sri Lanka. Coordinates: {lat:.4f}°N, {lng:.4f}°E.{disambig}"
    })
    col_pairs.append({
        "instruction": f"{name} thiyenne koheda?",
        "input": "",
        "output": f"{name} thiyenne {province} province eke, {district} district eke. Coordinates {lat:.4f}°N, {lng:.4f}°E. {disambig}"
    })
    col_pairs.append({
        "instruction": f"machan {name} thiyenne mona district ekedha?",
        "input": "",
        "output": f"{name} thiyෙන්නේ {district} district එකේ. {disambig}"
    })
    col_pairs.append({
        "instruction": f"how to go to {name} / {name} yanne kohomada?",
        "input": "",
        "output": f"{name} ({district}) yanna access road eka {road}. {budget['transport_en']}."
    })

    # ── 2. District/Province Q&A ─────────────────────────────────────────────
    si_pairs.append({
        "instruction": f"{name} කුමන දිස්ත්‍රික්කයේද?",
        "input": "",
        "output": f"{name} {district} දිස්ත්‍රික්කයේ, {province} පළාතේ ඇත.{disambig}"
    })
    en_pairs.append({
        "instruction": f"In which district is {name}?",
        "input": "",
        "output": f"{name} is in {district} District, {province} Province.{disambig}"
    })

    # ── 3. Description Q&A ───────────────────────────────────────────────────
    if desc and len(desc) > 10:
        si_pairs.append({
            "instruction": f"{name} ගැන කියන්න.",
            "input": "",
            "output": f"{desc} (ස්ථානය: {district} දිස්ත්‍රික්කය, {province} පළාත, ශ්‍රී ලංකාව)"
        })
        en_pairs.append({
            "instruction": f"Tell me about {name}.",
            "input": "",
            "output": f"{desc} (Location: {district} District, {province} Province, Sri Lanka)"
        })
        col_pairs.append({
            "instruction": f"explain about {name} / {name} kiyanne mokakda?",
            "input": "",
            "output": f"{name} kiyන්නේ {district} district eke ({province}) thiyena {category} place ekak. {desc}"
        })

    # ── 4. Best time / Activities Q&A ────────────────────────────────────────
    si_pairs.append({
        "instruction": f"{name} නැරඹීමට හොඳම කාලය කවදාද?",
        "input": "",
        "output": f"{name} ({district}, {province}) නැරඹීමට හොඳම කාලය: {best_time}. ක්‍රියාකාරකම්: {activities}."
    })
    en_pairs.append({
        "instruction": f"What is the best time to visit {name}?",
        "input": "",
        "output": f"The best time to visit {name} in {district} District is: {best_time}. Recommended activities: {activities}."
    })
    col_pairs.append({
        "instruction": f"best season eka mokakda {name} visit karanna?",
        "input": "",
        "output": f"{name} visit කරන්න හොඳම කාලය {best_time}. එහෙ ගියාම {activities} කරන්න පුළුවන්."
    })
    col_pairs.append({
        "instruction": f"{name} patte giyoth karanna puluwan deval monawada?",
        "input": "",
        "output": f"{name} ({district}) giyama karanna puluwan activities: {activities}."
    })

    # ── 5. Detailed Budget / Cost / Ticket Q&A ───────────────────────────────
    si_pairs.append({
        "instruction": f"{name} ප්‍රවේශ පත්‍රිකා මිල කොපමණද?",
        "input": "",
        "output": f"{name} ({district}) {budget['entry_fee_si']} ආරක්ෂාව: {safety}. මාර්ගය: {road}."
    })
    en_pairs.append({
        "instruction": f"What is the ticket price for {name}?",
        "input": "",
        "output": f"{budget['entry_fee_en']} for {name} ({district} District). Safety level: {safety}. Road condition: {road}."
    })
    col_pairs.append({
        "instruction": f"{name} ticket details and entry fee LKR walin kiyada?",
        "input": "",
        "output": (
            f"{name} yanna {budget['entry_fee_si']}. Budget estimates for LKR: "
            f"kema bim walata {budget['food_en']}. Transport cost eka {budget['transport_en']}. "
            f"Overall meka {budget['level_en']} category ekakata watenne."
        )
    })
    col_pairs.append({
        "instruction": f"budget trip ekakata {name} set da?",
        "input": "",
        "output": (
            f"ඔව්, {name} {budget['level_si']} චාරිකාවකි. "
            f"Entry Ticket: {budget['entry_fee_si']}. Food details: {budget['food_si']}. "
            f"Transport details: {budget['transport_si']}. Budget plans walata meka hondayi."
        )
    })
    col_pairs.append({
        "instruction": f"ලංකාවේ සල්ලි වලින් {name} යන්න කොච්චර වගේ යනවද?",
        "input": "",
        "output": (
            f"{name} නැරඹීමට {budget['entry_fee_si']} ප්‍රවාහනය සඳහා {budget['transport_si']} "
            f"සහ කෑම බීම සඳහා {budget['food_si']} අවශ්‍ය වේ. මෙය සමස්තයක් ලෙස {budget['level_si']} මට්ටමේ චාරිකාවකි."
        )
    })

    # ── 6. Popularity Q&A ────────────────────────────────────────────────────
    si_pairs.append({
        "instruction": f"{name} ජනප්‍රිය සංචාරක ගමනාන්තයක්ද?",
        "input": "",
        "output": f"{name} ({district}, {province}) {pop_si} ශ්‍රී ලංකා සංචාරක ගමනාන්තයකි."
    })
    en_pairs.append({
        "instruction": f"Is {name} a popular tourist destination?",
        "input": "",
        "output": f"{name} in {district} District is a {popularity.lower()} popularity tourist destination in Sri Lanka."
    })

    # ── 7. Multi-region spans Q&A (rivers, roads) ────────────────────────────
    if spans:
        spans_str = ", ".join(spans)
        si_pairs.append({
            "instruction": f"{name} කුමන දිස්ත්‍රික්ක හරහා ගමන් කරයිද?",
            "input": "",
            "output": f"{name} ශ්‍රී ලංකාවේ දිස්ත්‍රික්ක කිහිපයක් හරහා ගමන් කරයි: {spans_str}. ප්‍රදේශය අනුව ස්ථානීය නම් හා ලක්ෂණ වෙනස් විය හැකිය."
        })
        en_pairs.append({
            "instruction": f"Which districts does {name} pass through?",
            "input": "",
            "output": f"{name} passes through: {spans_str}. Local names and characteristics may vary by region."
        })

    # ── 8. Historical figure note (if applicable) ────────────────────────────
    if hist_note:
        si_pairs.append({
            "instruction": f"{name} ගැන ඉතිහාසය කියන්න.",
            "input": "",
            "output": f"{hist_note}"
        })
        en_pairs.append({
            "instruction": f"Tell me about the history of {name}.",
            "input": "",
            "output": f"{hist_note}"
        })

    # ── 9. Free entry Q&A ────────────────────────────────────────────────────
    si_pairs.append({
        "instruction": f"{name} නොමිලේ නැරඹිය හැකිද?",
        "input": "",
        "output": f"{'ඔව්, ' + name + ' නොමිලේ (Free) නැරඹිය හැකිය.' if budget['is_free'] else name + ' සඳහා ' + budget['entry_fee_si']} ({district} දිස්ත්‍රික්කය)"
    })
    en_pairs.append({
        "instruction": f"Is {name} free to visit?",
        "input": "",
        "output": f"{'Yes, ' + name + ' is free to enter.' if budget['is_free'] else 'No, ' + budget['entry_fee_en']} ({district} District)"
    })

    # ── 10. Family-friendly Q&A ──────────────────────────────────────────────
    si_pairs.append({
        "instruction": f"{name} දරුවන් සමඟ යාමට සුදුසුද?",
        "input": "",
        "output": f"{'ඔව්, ' + name + ' ළමයින් සහ පවුලේ සාමාජිකයන් සමඟ යාමට සුදුසු ස්ථානයකි.' if is_family else name + ' දරුවන් සමඟ යාමට ප්‍රවේශම් විය යුතු ස්ථානයකි.'} ({district}, {province}). ආරක්ෂාව: {safety}."
    })
    en_pairs.append({
        "instruction": f"Is {name} family-friendly?",
        "input": "",
        "output": f"{'Yes, ' + name + ' is family-friendly and suitable for children.' if is_family else name + ' requires caution for young children.'} Safety level: {safety}. ({district} District)"
    })
    col_pairs.append({
        "instruction": f"kids ekka {name} yanna puluwanda?",
        "input": "",
        "output": f"{'ඔව්, ' + name + ' family / kids friendly.' if is_family else name + ' yaddi poddak caution වෙන්න. Safety level එක: ' + safety}."
    })

    # ── 11. Safety / Monsoon Q&A ─────────────────────────────────────────────
    if monsoon and monsoon.lower() != "none":
        si_pairs.append({
            "instruction": f"{name} වැසි කාලයේ යාමට ආරක්ෂිතද?",
            "input": "",
            "output": f"{name} ({district}): {rain}. {monsoon}. ආරක්ෂා මට්ටම: {safety}."
        })
        en_pairs.append({
            "instruction": f"Is it safe to visit {name} during monsoon?",
            "input": "",
            "output": f"{name} ({district}): Rain sensitivity — {rain}. Note: {monsoon}. Safety: {safety}."
        })
        col_pairs.append({
            "instruction": f"vassa kaleta {name} yanna safe da?",
            "input": "",
            "output": f"{name} වැසි කාලයට yaddi {rain_sensitivity_col(rain)}. Note: {monsoon}. Safety is {safety}."
        })

    # ── 12. Complete travel summary Q&A ──────────────────────────────────────
    si_pairs.append({
        "instruction": f"{name} ගැන සම්පූර්ණ සංචාරක තොරතුරු දෙන්න.",
        "input": "",
        "output": (
            f"📍 {name}\n"
            f"📌 ස්ථානය: {district} දිස්ත්‍රික්කය, {province} පළාත, ශ්‍රී ලංකාව\n"
            f"🏷️ වර්ගය: {category}\n"
            f"📝 විස්තර: {desc}\n"
            f"📅 හොඳම කාලය: {best_time}\n"
            f"🎯 ක්‍රියාකාරකම්: {activities}\n"
            f"🎟️ ගාස්තු: {budget['entry_fee_si']}\n"
            f"🛡️ ආරක්ෂාව: {safety}\n"
            f"🚗 මාර්ග: {road}\n"
            f"⭐ ජනප්‍රියතාව: {popularity}"
        )
    })
    en_pairs.append({
        "instruction": f"Give me complete travel information about {name}.",
        "input": "",
        "output": (
            f"📍 {name}\n"
            f"📌 Location: {district} District, {province} Province, Sri Lanka\n"
            f"🏷️ Category: {category}\n"
            f"📝 About: {desc}\n"
            f"📅 Best Time: {best_time}\n"
            f"🎯 Activities: {activities}\n"
            f"🎟️ Entry Fee: {budget['entry_fee_en']}\n"
            f"🛡️ Safety: {safety}\n"
            f"🚗 Road Access: {road}\n"
            f"⭐ Popularity: {popularity}"
        )
    })
    # ── 13. Category-Specific Expert Q&A ─────────────────────────────────────
    category_lower = category.lower()
    if "religious" in category_lower or "temple" in category_lower:
        col_pairs.append({
            "instruction": f"dress code eka kohomada {name} yaddi?",
            "input": "",
            "output": f"ශ්‍රී ලංකාවේ ආගමික ස්ථාන වන {name} වැනි තැන්වලට යද්දී ශරීරය වැසෙන සේ චාරිත්‍රානුකූල ශිෂ්ට ඇඳුම් ඇඳීම අවශ්‍ය වේ (සුදු පැහැති ඇඳුම් වඩාත් සුදුසුය). පාවහන් සහ හිස්වැසුම් ඇතුළට යාමට පෙර ගැලවිය යුතුය."
        })
        en_pairs.append({
            "instruction": f"What is the dress code for {name}?",
            "input": "",
            "output": f"When visiting religious places like {name}, please wear modest clothing covering shoulders and knees (preferably white). Remove hats and shoes before entering."
        })
        col_pairs.append({
            "instruction": f"{name} athule photo ganna puluwanda?",
            "input": "",
            "output": "බොහෝ ආගමික ස්ථාන වල පිළිම හෝ පූජනීය තැන් ඡායාරූපගත කිරීම සීමා කර ඇත. පිළිමවලට පිටුපස හරවා සෙල්ෆි/ඡායාරූප ගැනීම සපුරා තහනම්ය."
        })

    elif "water-falls" in category_lower or "waterfall" in category_lower:
        col_pairs.append({
            "instruction": f"{name} eken nanna (swim කරන්න) safe da?",
            "input": "",
            "output": f"{name} දිය ඇල්ලෙන් නෑම ඉතා අවදානම් විය හැක. ආරක්ෂිත මට්ටම: {safety}. ගල් ලිස්සන සුළු වන අතර, වැසි කාලයේදී ක්ෂණිකව ජල මට්ටම ඉහළ යා හැක. ආරක්ෂිතම ස්ථාන පමණක් තෝරාගන්න."
        })
        en_pairs.append({
            "instruction": f"Can we swim or bathe at {name}?",
            "input": "",
            "output": f"Swimming or bathing at {name} can be risky. Safety level is {safety}. Rocks are slippery, and sudden water level rises are common during rains. Exercise extreme caution."
        })
        col_pairs.append({
            "instruction": f"{name} laga koodallo (leeches) innawada?",
            "input": "",
            "output": f"{name} දියඇල්ල සහ ඒ අවට තෙත් වනාන්තර ආශ්‍රිතව කූඩැල්ලන් (leeches) සිටීමේ ප්‍රවණතාව වැඩිය. Leech repellent හෝ ලුණු වතුර රැගෙන යාම සුදුසුය."
        })

    elif "beach" in category_lower:
        col_pairs.append({
            "instruction": f"{name} beach eke swim කරන්න safe da?",
            "input": "",
            "output": f"{name} මුහුදු තීරයේ පිහිනීම සාමාන්‍යයෙන් {safety} වේ. එහෙත් වාරකන් (monsoon) කාලයේදී මුහුද රළු විය හැකි බැවින් ප්‍රවේශම් වන්න."
        })
        col_pairs.append({
            "instruction": f"{name} eke sunset/sunrise balanna hoda welawa kiyada?",
            "input": "",
            "output": f"{name} වෙරළේ sunset (හිරු බැසීම) නැරඹීමට සවස 5.30 - 6.30 ත්, sunrise (හිරු උදාව) සඳහා උදේ 5.30 - 6.15 ත් වඩාත් සුදුසු වේ."
        })

    elif "adventure" in category_lower:
        col_pairs.append({
            "instruction": f"{name} hike karanna yaddi mon wගේ ඇඳුම්ද අඳින්න ඕනේ?",
            "input": "",
            "output": f"{name} හි ක්‍රියාකාරකම් වලට සහභාගී වීමේදී පහසු ක්‍රීඩා ඇඳුම් (sports clothing) සහ හොඳ ග්‍රිප් එකක් සහිත සපත්තු ඇඳීම නිර්දේශ කෙරේ."
        })

    elif "view" in category_lower or "point" in category_lower:
        col_pairs.append({
            "instruction": f"{name} eke drone fly karanna puluwanda?",
            "input": "",
            "output": f"{name} හි ඩ්‍රෝන (drones) පියාසර කිරීමට පෙර ශ්‍රී ලංකා සිවිල් ගුවන් සේවා අධිකාරියේ (CAA) සහ අදාළ ප්‍රාදේශීය වනජීවී/ආරක්ෂක අංශවල නීත්‍යානුකූල අවසරය ලබාගත යුතුය."
        })

    return si_pairs, en_pairs, col_pairs


def rain_sensitivity_col(rain_str: str) -> str:
    """Helper to localize rain sensitivity for Singlish."""
    if "high" in rain_str.lower() or "unsafe" in rain_str.lower():
        return "godak risk/danger thiyenawa (not safe)"
    return "poddak parissamin yanna puluwan (safe enough)"


def generate_aggregate_qa_pairs(records: list) -> list:
    """
    Generates multi-record aggregate Q&A pairs (e.g. recommendations by district/category, free places).
    """
    by_district = {}
    by_category = {}
    
    for rec in records:
        dist = str(rec.get("district_id", rec.get("district", "Unknown"))).strip()
        cat = str(rec.get("category_id", rec.get("category", "Place"))).strip()
        
        if dist not in by_district:
            by_district[dist] = []
        by_district[dist].append(rec)
        
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(rec)
        
    agg_pairs = []
    
    # 1. District recommendations
    for dist, dist_recs in by_district.items():
        if len(dist_recs) >= 3:
            sorted_recs = sorted(dist_recs, key=lambda x: 0 if str(x.get("tourist_popularity", "Medium")).strip() == "High" else (1 if str(x.get("tourist_popularity", "Medium")).strip() == "Medium" else 2))
            top_places = sorted_recs[:5]
            names_str = ", ".join([r.get("name", "Unknown") for r in top_places])
            
            agg_pairs.append({
                "instruction": f"{dist} දිස්ත්‍රික්කයේ බලන්න යන්න හොඳම ස්ථාන මොනවාද?",
                "input": "",
                "output": f"{dist} දිස්ත්‍රික්කයේ නැරඹීමට ඇති හොඳම ස්ථාන කිහිපයක් මෙන්න: {names_str}. මේවායේ විස්තර සහ පිහිටීම් වෙන වෙනම Lumen-1 වෙතින් ලබා ගත හැක."
            })
            
            agg_pairs.append({
                "instruction": f"What are the best places to visit in {dist} district?",
                "input": "",
                "output": f"Here are some of the best places to visit in {dist} District: {names_str}. You can query each place individually for maps, fees, and safety guidelines."
            })
            
            agg_pairs.append({
                "instruction": f"{dist} eke visit karanna thiyena best places monawada?",
                "input": "",
                "output": f"{dist} district eke yanna hoda top places tikak meka: {names_str}. Oya places gana detail information (ticket rates, travel guide) separate ahanna puluwan."
            })
            
    # 2. Category recommendations
    for cat, cat_recs in by_category.items():
        if len(cat_recs) >= 3:
            sorted_recs = sorted(cat_recs, key=lambda x: 0 if str(x.get("tourist_popularity", "Medium")).strip() == "High" else (1 if str(x.get("tourist_popularity", "Medium")).strip() == "Medium" else 2))
            top_places = sorted_recs[:5]
            names_str_en = ", ".join([f"{r.get('name', 'Unknown')} ({r.get('district', 'Unknown')} District)" for r in top_places])
            
            agg_pairs.append({
                "instruction": f"Sri Lanka eke thiyena popular {cat} places monawada?",
                "input": "",
                "output": f"Sri Lanka eke thiyena popularම {cat} list ekak mehema thiyenawa: {names_str_en}. Godak kattiya me places walata visit karanawa."
            })
            
    # 3. Free/Budget places by district
    for dist, dist_recs in by_district.items():
        free_places = [r for r in dist_recs if "free" in str(r.get("ticket_price", "Free")).lower() or str(r.get("ticket_price")) == "0"]
        if len(free_places) >= 2:
            names_str = ", ".join([r.get("name", "Unknown") for r in free_places[:5]])
            agg_pairs.append({
                "instruction": f"{dist} wala salli epa wenne nathi (free) places monawada?",
                "input": "",
                "output": f"{dist} district eke entry tickets nathi (free of charge) yanna puluwan places: {names_str}. Budget trip yanna honda than mewa."
            })
            agg_pairs.append({
                "instruction": f"නොමිලේ නැරඹිය හැකි ස්ථාන මොනවාද {dist} දිස්ත්‍රික්කයේ?",
                "input": "",
                "output": f"{dist} දිස්ත්‍රික්කයේ ප්‍රවේශ ගාස්තු රහිතව නැරඹීමට යා හැකි ස්ථාන කිහිපයක් මෙන්න: {names_str}."
            })

    return agg_pairs




# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Mount Google Drive (if Colab)
# ─────────────────────────────────────────────────────────────────────────────
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("✅ Google Drive mounted at /content/drive")
    except Exception as e:
        print(f"⚠️ Could not mount Drive: {e}")
        print("   Trying to find files in current directory (/content)...")
        GDRIVE_BASE = "/content"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Find and load all JSON files
# ─────────────────────────────────────────────────────────────────────────────
print("\n📂 Loading JSON files...")
all_records = []
seen_ids    = set()   # For deduplication by record id
seen_names  = set()   # For deduplication by name

for filename in JSON_FILES:
    # Search in multiple possible locations
    search_paths = [
        os.path.join(GDRIVE_BASE, filename),
        os.path.join("/content", filename),
        filename,  # current directory
    ]
    
    found_path = None
    for path in search_paths:
        if os.path.exists(path):
            found_path = path
            break
    
    if not found_path:
        print(f"⚠️ Not found: {filename} — skipping.")
        continue
    
    try:
        with open(found_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list [] and dict {"data": [...]} formats
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Try common wrapper keys
            records = data.get("data", data.get("places", data.get("locations", [])))
        else:
            records = []
        
        added = 0
        for rec in records:
            # Deduplication: skip if same id or same name already seen
            rec_id   = rec.get("id", "")
            rec_name = str(rec.get("name", "")).strip().lower()
            
            if rec_id and rec_id in seen_ids:
                continue
            if rec_name and rec_name in seen_names:
                continue
            
            # Ensure id exists
            if not rec_id:
                rec["id"] = f"pl_{hashlib.md5(rec_name.encode('utf-8')).hexdigest()[:8]}"
            
            seen_ids.add(rec.get("id", ""))
            seen_names.add(rec_name)
            all_records.append(rec)
            added += 1
        
        print(f"   ✅ {filename}: {added} unique records loaded ({len(records)} total in file)")
    
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse error in {filename}: {e}")
    except Exception as e:
        print(f"   ❌ Error loading {filename}: {e}")

print(f"\n📊 Total unique records loaded: {len(all_records)}")

if len(all_records) == 0:
    print("❌ No records found. Please check file names and paths.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Save merged JSON
# ─────────────────────────────────────────────────────────────────────────────
with open(OUTPUT_MERGED_JSON, "w", encoding="utf-8") as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)
print(f"💾 Merged JSON saved: {OUTPUT_MERGED_JSON} ({len(all_records)} records)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Generate SFT Q&A pairs
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n🧠 Generating disambiguation-safe SFT training pairs...")

import time
start_gen_time = time.time()
# 11 hours 45 minutes = 42300 seconds
TIMEOUT_SECONDS = 11 * 3600 + 45 * 60
timeout_triggered = False

all_si_pairs = []
all_en_pairs = []
all_mixed    = []
processed_count = 0

for i, record in enumerate(all_records):
    # Check if approaching environment timeout
    if time.time() - start_gen_time > TIMEOUT_SECONDS:
        print(f"\n⏳ Timeout Warning: Execution has run for 11 hours 45 minutes.")
        print(f"   Exiting early to save generated pairs ({len(all_mixed)}) and trigger download...")
        timeout_triggered = True
        break
        
    si, en, col = generate_qa_pairs(record)
    all_si_pairs.extend(si)
    all_en_pairs.extend(en)
    all_mixed.extend(si + en + col)
    processed_count = i + 1
    
    # Progress every 500 records
    if (i + 1) % 500 == 0:
        print(f"   Processed {i+1}/{len(all_records)} records...")

print("\n📦 Generating multi-record recommendations and aggregates...")
processed_records = all_records[:processed_count] if timeout_triggered else all_records
agg_pairs = generate_aggregate_qa_pairs(processed_records)
all_mixed.extend(agg_pairs)

print(f"✅ Q&A generation complete!")
print(f"   Sinhala pairs    : {len(all_si_pairs)}")
print(f"   English pairs    : {len(all_en_pairs)}")
print(f"   Colloquial/Mixed : {len(all_mixed) - len(all_si_pairs) - len(all_en_pairs) - len(agg_pairs)}")
print(f"   Aggregate pairs  : {len(agg_pairs)}")
print(f"   Total SFT pairs  : {len(all_mixed)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Write JSONL files
# ─────────────────────────────────────────────────────────────────────────────
def write_jsonl(filepath: str, records: list):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            # Create a copy to avoid mutating original dict if shared
            output_rec = dict(rec)
            # Add ChatML/messages conversational format for 100% SFTTrainer compatibility
            output_rec["messages"] = [
                {"role": "user", "content": rec["instruction"]},
                {"role": "assistant", "content": rec["output"]}
            ]
            f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
    print(f"💾 Saved: {filepath} ({len(records)} pairs)")

write_jsonl(OUTPUT_SFT_ALL,     all_mixed)
write_jsonl(OUTPUT_SFT_SINHALA, all_si_pairs)
write_jsonl(OUTPUT_SFT_ENGLISH, all_en_pairs)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Preview sample records
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📋 SAMPLE Q&A PAIRS (first 3):")
print("="*60)
for sample in all_mixed[:3]:
    print(f"\n  Q: {sample['instruction']}")
    print(f"  A: {sample['output'][:120]}...")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Save / Download output files
# ─────────────────────────────────────────────────────────────────────────────
if IN_KAGGLE:
    # Kaggle: /kaggle/working/ folder eke save karanawa (auto-available for download)
    import shutil
    kaggle_out = "/kaggle/working"
    for src in [OUTPUT_MERGED_JSON, OUTPUT_SFT_ALL, OUTPUT_SFT_SINHALA, OUTPUT_SFT_ENGLISH]:
        if os.path.exists(src):
            dst = os.path.join(kaggle_out, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy(src, dst)
                print(f"✅ Saved to Kaggle output: {dst}")
            else:
                print(f"✅ Already generated in Kaggle output: {dst}")
    print("\n📥 Kaggle notebook right panel → Output tab → Files download කරන්න.")

elif IN_COLAB:
    try:
        from google.colab import files
        print("\n📥 Downloading output files to your computer...")
        files.download(OUTPUT_MERGED_JSON)
        files.download(OUTPUT_SFT_ALL)
        files.download(OUTPUT_SFT_SINHALA)
        files.download(OUTPUT_SFT_ENGLISH)
        print("✅ Downloads triggered!")
    except Exception as e:
        print(f"⚠️ Auto-download failed: {e}")
        print("   Left panel → Files icon → Manually download.")

else:
    print(f"\n💻 Output files saved locally:")
    for f_path in [OUTPUT_MERGED_JSON, OUTPUT_SFT_ALL, OUTPUT_SFT_SINHALA, OUTPUT_SFT_ENGLISH]:
        if os.path.exists(f_path):
            print(f"   📄 {os.path.abspath(f_path)}")

print("\n🎉 Done! Your SFT training data is ready.")
