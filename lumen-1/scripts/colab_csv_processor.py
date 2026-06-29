import os
import sys
import json
import re
import hashlib
import uuid
import subprocess
import time
import urllib.request

# --- Google Colab Dependencies Setup ---
def check_and_install_dependencies():
    """
    Google Colab පරිසරයේදී අවශ්‍ය packages ස්වයංක්‍රීයව install කරයි.
    Installs packages automatically if running in Google Colab.
    """
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        print("🌐 Google Colab environment detected. Installing required packages...")
        # Colab requires bitsandbytes, accelerate, openpyxl, pandas, osmiter (pure python)
        packages = ["transformers", "accelerate", "bitsandbytes", "pandas", "openpyxl", "sentencepiece", "osmiter"]
        for pkg in packages:
            try:
                __import__(pkg)
            except ImportError:
                print(f"📦 Installing {pkg}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print("✅ All dependencies installed successfully!")
    else:
        print("💻 Local execution environment detected. Skipping package installation.")

# Run package installation checks
check_and_install_dependencies()

# Import pandas and torch after dependency checks
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Try importing osmiter for PBF parsing
try:
    import osmiter
except ImportError:
    osmiter = None

# --- Configuration ---
# Google Colab එකේදී run වෙනකොට පාවිච්චි කරන්න පුළුවන් හොඳම model එකක් තමයි Qwen 2.5
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" 
OUTPUT_JSON_PATH = "tripme_database_complete_NEW.json"
OUTPUT_CSV_PATH  = "tripme_database_complete_NEW.csv"
OUTPUT_SFT_PATH  = "tripme_sft_training.jsonl"  # SFT training Q&A pairs (disambiguation-safe)

# OSM PBF settings
OSM_PBF_MODE = True  # Set to True to extract data from Geofabrik Sri Lanka PBF
OSM_PBF_URL = "https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf"
OSM_PBF_PATH = "sri-lanka-latest.osm.pbf"
OSM_LIMIT_LOCATIONS = 15000  # Extract 10,000 - 15,000+ locations
OSM_FAST_HEURISTIC_MODE = True  # Uses fast rule-based engine to complete 15k spots in 5 seconds

# If OSM_PBF_MODE is False, this CSV/Excel file path will be used:
INPUT_FILE_PATH = None

# Colab runtime limit in seconds (පැය 1යි විනාඩි 40කින් පසු ඉබේම නතර වී download වීමට)
MAX_RUN_TIME_SECONDS = 1 * 3600 + 40 * 60

# --- Sri Lanka District Coordinates Center Mapping ---
DISTRICT_CENTERS = {
    "Colombo": (6.9271, 79.8612, "Western"),
    "Gampaha": (7.0873, 80.0144, "Western"),
    "Kalutara": (6.5854, 79.9607, "Western"),
    "Kandy": (7.2906, 80.6337, "Central"),
    "Matale": (7.4675, 80.6234, "Central"),
    "Nuwara Eliya": (6.9497, 80.7891, "Central"),
    "Galle": (6.0535, 80.2210, "Southern"),
    "Matara": (5.9549, 80.5550, "Southern"),
    "Hambantota": (6.1248, 81.1185, "Southern"),
    "Jaffna": (9.6615, 80.0255, "Northern"),
    "Kilinochchi": (9.3803, 80.3992, "Northern"),
    "Mannar": (8.9810, 79.9044, "Northern"),
    "Vavuniya": (8.7542, 80.4982, "Northern"),
    "Mullaitivu": (9.2673, 80.8143, "Northern"),
    "Batticaloa": (7.7170, 81.7010, "Eastern"),
    "Ampara": (7.2912, 81.6724, "Eastern"),
    "Trincomalee": (8.5874, 81.2152, "Eastern"),
    "Kurunegala": (7.4863, 80.3647, "North Western"),
    "Puttalam": (8.0362, 79.8283, "North Western"),
    "Anuradhapura": (8.3114, 80.4037, "North Central"),
    "Polonnaruwa": (7.9403, 81.0000, "North Central"),
    "Badulla": (6.9934, 81.0550, "Uva"),
    "Moneragala": (6.8724, 81.3507, "Uva"),
    "Ratnapura": (6.6828, 80.3992, "Sabaragamuwa"),
    "Kegalle": (7.2513, 80.3464, "Sabaragamuwa")
}

def resolve_district_province(lat, lng):
    """
    GPS coordinates ආශ්‍රයෙන් ආසන්නතම දිස්ත්‍රික්කය සහ පළාත තීරණය කරයි.
    Finds the closest district and province based on latitude and longitude distance.
    """
    min_dist = float('inf')
    best_district = "Unknown"
    best_province = "Unknown"
    for dist, (d_lat, d_lng, prov) in DISTRICT_CENTERS.items():
        distance = (lat - d_lat)**2 + (lng - d_lng)**2
        if distance < min_dist:
            min_dist = distance
            best_district = dist
            best_province = prov
    return best_district, best_province

# --- Heuristic Data Compilation (Fast Rule-based metadata generator) ---
def generate_heuristic_description(name, category, district, province):
    templates = [
        f"{name} is a beautiful {category.lower()} located in the {district} district of the {province} province, Sri Lanka. It represents a highly recommended stop for travelers interested in exploring the unique local charm and sightseeing spots of the region.",
        f"Located in {district} ({province} Province), {name} is a popular {category.lower()} offering visitors a glimpse into Sri Lanka's local culture, breathtaking views, and rich natural surroundings.",
        f"{name} is a scenic {category.lower()} in the {district} district, Sri Lanka. Known for its pleasant climate and visual appeal, it is a perfect spot for photography, adventure, and relaxation.",
        f"A beautiful landmark found in {district}, {province} Province, {name} stands as an important {category.lower()} attraction. It is a must-visit location for anyone exploring the travel highlights of Sri Lanka."
    ]
    idx = sum(ord(c) for c in name) % len(templates)
    return templates[idx]

def get_heuristic_metadata(name, category, district, province, lat, lng, opening_hours_osm):
    category_lower = category.lower()
    
    # Standard Defaults
    best_time_to_visit = "December to March"
    ticket_price = "Free"
    parking_avail = "yes"
    toilets = "yes"
    food_nearby = "yes"
    wheelchair_access = "no"
    camping_allowed = "no"
    safety_level = "Safe"
    wildlife_hazard = "None"
    guide_required = "no"
    rain_sensitivity = "Safe"
    monsoon_note = "None"
    opening_hours = opening_hours_osm if opening_hours_osm and opening_hours_osm != "Unknown" else "24 Hours"
    mobile_signal = "Good"
    road_condition = "Paved"
    activities = "Sightseeing, Photography"
    tourist_popularity = "Medium"
    budget_category = "Free"
    height_m = 0
    length_km = 0
    
    # Apply category logic rules
    if "beach" in category_lower or category_lower == "beach":
        best_time_to_visit = "December to April" if lat < 7.5 else "May to September"
        activities = "Swimming, Sunbathing, Photography, Sunset Watching"
        tourist_popularity = "High"
        budget_category = "Free"
        
    elif "peak" in category_lower or "mountain" in category_lower or category_lower in ["hiking", "volcano", "rock", "ridge"]:
        best_time_to_visit = "January to April"
        parking_avail = "no"
        toilets = "no"
        food_nearby = "no"
        camping_allowed = "yes"
        safety_level = "Moderate"
        road_condition = "Trekking"
        activities = "Hiking, Trekking, Sunrise Watching, Photography"
        tourist_popularity = "Medium"
        
    elif "waterfall" in category_lower or "water" in category_lower or category_lower == "cascade":
        best_time_to_visit = "September to January"
        parking_avail = "no"
        toilets = "no"
        safety_level = "Moderate"
        rain_sensitivity = "Dangerous during heavy rain"
        activities = "Sightseeing, Swimming, Photography"
        tourist_popularity = "High"
        
    elif "worship" in category_lower or category_lower in ["temple", "monastery", "church", "mosque", "kovil"]:
        best_time_to_visit = "All year round"
        ticket_price = "Free (Donations welcome)"
        activities = "Cultural Exploration, Sightseeing, Worship"
        tourist_popularity = "High"
        
    elif category_lower in ["nature_reserve", "national_park", "zoo", "aquarium", "forest"]:
        best_time_to_visit = "December to April"
        ticket_price = "2500 LKR"
        safety_level = "Moderate"
        wildlife_hazard = "Elephant and wild animal encounters possible"
        road_condition = "4WD Required"
        activities = "Wildlife Safari, Photography, Bird Watching"
        tourist_popularity = "High"
        budget_category = "Moderate"
        
    elif category_lower in ["restaurant", "cafe", "bar", "fast_food"]:
        budget_category = "Budget"
        activities = "Dining, Socializing"
        
    # Popularity keywords override
    popular_keywords = ["sigiriya", "ella", "galle", "kandy", "yala", "udawalawe", "mirissa", "hikkaduwa", "unawatuna", "pinnawala", "adams peak", "sripada", "horton plains"]
    name_lower = name.lower()
    if any(kw in name_lower for kw in popular_keywords):
        tourist_popularity = "High"
        if "sigiriya" in name_lower or "galle fort" in name_lower:
            ticket_price = "4500 LKR"
            budget_category = "Expensive"

    return {
        "description": generate_heuristic_description(name, category, district, province),
        "best_time_to_visit": best_time_to_visit,
        "ticket_price": ticket_price,
        "parking_avail": parking_avail,
        "toilets": toilets,
        "food_nearby": food_nearby,
        "wheelchair_access": wheelchair_access,
        "camping_allowed": camping_allowed,
        "safety_level": safety_level,
        "wildlife_hazard": wildlife_hazard,
        "guide_required": guide_required,
        "rain_sensitivity": rain_sensitivity,
        "monsoon_note": monsoon_note,
        # Use consistent lat/lng naming matching LLM mode output
        "lat": lat,
        "lng": lng,
        "opening_hours": opening_hours,
        "mobile_signal": mobile_signal,
        "road_condition": road_condition,
        "activities": activities,
        "tourist_popularity": tourist_popularity,
        "budget_category": budget_category,
        # Use consistent casing matching LLM mode output
        "Height_m": str(height_m),
        "Length_km": str(length_km)
    }

# --- Model setup ---
def setup_model():
    """
    Loads tokenizer and model in 4-bit mode for memory efficiency.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"🚀 Loading tokenizer and model: {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using Device: {device.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto"
        )
    else:
        print("⚠️ WARNING: GPU not found. Running on CPU (This will be SLOW!).")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto"
        )
        
    return tokenizer, model, device

def generate_and_verify(tokenizer, model, device, place_name, existing_data_dict):
    """
    Queries the LLM to generate completed profiles with exact fields.
    """
    existing_data_str = json.dumps(existing_data_dict, ensure_ascii=False)
    
    prompt = f"""You are an expert Sri Lankan travel guide and data compiler.
Place: {place_name}
Existing Data from CSV: {existing_data_str}

Task: You must generate a complete JSON profile for this place. 
Use your knowledge to fill in EVERY field with realistic and accurate details. 

Return strictly ONLY a valid JSON object matching this exact format:
{{
    "description": "Write a highly specific 50-word rich description here",
    "best_time_to_visit": "e.g., Dec to April",
    "ticket_price": "Free or amount in LKR",
    "parking_avail": "yes or no",
    "toilets": "yes or no",
    "food_nearby": "yes or no",
    "wheelchair_access": "yes or no",
    "camping_allowed": "yes or no",
    "safety_level": "Safe, Moderate, or Dangerous",
    "wildlife_hazard": "Explain any animal hazards or say None",
    "guide_required": "yes or no",
    "rain_sensitivity": "Dangerous during heavy rain or Safe",
    "monsoon_note": "Avoid May-Sept or None",
    "latitude": "Provide the exact GPS latitude as a number e.g. 6.8291",
    "longitude": "Provide the exact GPS longitude as a number e.g. 79.8623",
    "opening_hours": "e.g., 24 Hours, or 8 AM - 5 PM",
    "mobile_signal": "Good, Weak, or None",
    "road_condition": "Paved, 4WD Required, or Trekking",
    "activities": "List 3-4 activities e.g., Swimming, Hiking, Photography",
    "tourist_popularity": "Classify popularity as: High, Medium, or Low",
    "budget_category": "Free, Budget, Moderate, or Expensive",
    "height_m": "Provide height in meters if it's a waterfall or mountain, else 0",
    "length_km": "Provide length in km if it's a river or beach, else 0"
}}
"""

    messages = [
        {"role": "system", "content": "You are a helpful travel assistant that outputs valid JSON profiles."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=800, temperature=0.2)
        
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    def extract_json_fields_fallback(malformed_str):
        extracted = {}
        string_patterns = [
            "description", "best_time_to_visit", "ticket_price", "parking_avail", 
            "toilets", "food_nearby", "wheelchair_access", "camping_allowed", 
            "safety_level", "wildlife_hazard", "guide_required", "rain_sensitivity", 
            "monsoon_note", "opening_hours", "mobile_signal", "road_condition", 
            "activities", "tourist_popularity", "budget_category"
        ]
        
        for key in string_patterns:
            pattern = re.compile(rf'"{key}"\s*:\s*"(.*?)"', re.DOTALL | re.IGNORECASE)
            match = pattern.search(malformed_str)
            if match:
                extracted[key] = match.group(1).replace('\\"', '"').replace('\n', ' ').strip()
                
        numeric_patterns = ["latitude", "longitude", "height_m", "length_km"]
        for key in numeric_patterns:
            pattern = re.compile(rf'"{key}"\s*:\s*([+-]?\d*\.\d+|[+-]?\d+)', re.IGNORECASE)
            match = pattern.search(malformed_str)
            if match:
                val_str = match.group(1)
                extracted[key] = float(val_str) if "." in val_str else int(val_str)
                
        if len(extracted) >= 3:
            return extracted
        return None

    def clean_and_parse_json(response_str):
        cleaned = re.sub(r"```json\s*", "", response_str)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()
        
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        
        if json_start == -1 or json_end == -1:
            return None
            
        json_str = cleaned[json_start:json_end]
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*\]", "]", json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return extract_json_fields_fallback(json_str)

    return clean_and_parse_json(response)


# =============================================================================
# --- Disambiguation Q&A Generator ---
# Eka record ekata training-ready Q&A pairs generate karanawa.
# Problem 1: Place name ambiguity  (e.g. "Nuwara" vs "Nuwara Eliya")
# Problem 2: Multi-region entities (rivers, roads spanning districts)
# Problem 3: Similar names         (e.g. Parakramabahu I vs II)
# =============================================================================

# Known multi-region entities - river/road names that span multiple districts.
# Format: { "name_keyword": ["District1", "District2", ...] }
MULTI_REGION_ENTITIES = {
    "mahaweli": ["Nuwara Eliya", "Kandy", "Matale", "Polonnaruwa", "Trincomalee"],
    "kelani":   ["Nuwara Eliya", "Kegalle", "Colombo"],
    "kalu":     ["Ratnapura", "Kalutara"],
    "gin":      ["Ratnapura", "Galle"],
    "walawe":   ["Ratnapura", "Hambantota"],
    "deduru":   ["Kurunegala", "Puttalam"],
    "a9":       ["Colombo", "Gampaha", "Kurunegala", "Anuradhapura", "Vavuniya", "Kilinochchi", "Jaffna"],
    "a1":       ["Colombo", "Kalutara", "Galle", "Matara", "Hambantota"],
}

# Known disambiguation pairs - places whose names are similar/confusing.
# Format: { "ambiguous_name": "disambiguation note in Sinhala+English" }
AMBIGUOUS_PLACE_NOTES = {
    "kandy lake":     "Kandy Lake (නුවර වැව) is in Kandy city, NOT Nuwara Eliya. Do not confuse with any lake in Nuwara Eliya.",
    "නුවර වැව":      "නුවර වැව = Kandy Lake. මහනුවර (Kandy) දිස්ත්‍රික්කයේ ය. නුවරඑළිය (Nuwara Eliya) නොවේ.",
    "galle":          "Galle city is in Galle District, Southern Province. Not to be confused with Galle Face in Colombo.",
    "galle face":     "Galle Face Green is in Colombo, Colombo District. Not in Galle city or Galle District.",
    "victoria":       "Victoria Reservoir is in Kandy District. Victoria Park is in Nuwara Eliya. They are different.",
    "gregory":        "Lake Gregory is in Nuwara Eliya District. Not the same as Kandy Lake.",
    "sigiriya":       "Sigiriya Rock Fortress is in Matale District, not Polonnaruwa District.",
    "anuradhapura":   "Anuradhapura is a district AND a city. The ancient lakes (Tissa Wewa, Nuwara Wewa) are within Anuradhapura city.",
}

def detect_multi_region(name: str):
    """
    Checks if a place name matches a known multi-region entity (river, road).
    Returns a list of districts it spans, or empty list if not multi-region.
    """
    name_lower = name.lower()
    for keyword, districts in MULTI_REGION_ENTITIES.items():
        if keyword in name_lower:
            return districts
    return []

def generate_disambiguation_qa(record: dict) -> list:
    """
    Eka location record ekata disambiguation-safe SFT Q&A pairs list ekak generate karanawa.
    
    Returns a list of {"instruction", "input", "output"} dicts (ChatML / Alpaca format).
    """
    name      = record.get("name", "Unknown")
    district  = record.get("district_id", record.get("district", "Unknown"))
    province  = record.get("province_id", record.get("province", "Unknown"))
    category  = record.get("category_id", record.get("category", "Place"))
    lat       = record.get("lat", 0.0)
    lng       = record.get("lng", 0.0)
    desc      = record.get("description", "")
    best_time = record.get("best_time_to_visit", "All year round")
    activities= record.get("activities", "Sightseeing")
    popularity= record.get("tourist_popularity", "Medium")
    
    qa_pairs = []
    name_lower = name.lower()
    
    # --- Ambiguity note (check known confusing places) ---
    disambig_suffix = ""
    for key, note in AMBIGUOUS_PLACE_NOTES.items():
        if key in name_lower:
            disambig_suffix = f" ⚠️ Disambiguation: {note}"
            break
    
    # --- Multi-region check ---
    spans = detect_multi_region(name)
    if spans:
        spans_str_en = ", ".join(spans)
        spans_str_si = ", ".join(spans)  # Use English district names (standard)
        
        qa_pairs.append({
            "instruction": f"{name} කොහේ ඉඳල ගලා යයිද / ගම්‍යද?",
            "input": "",
            "output": (
                f"{name} ශ්‍රී ලංකාවේ දිස්ත්‍රික්ක කිහිපයක් හරහා ගමන් කරයි: {spans_str_si}. "
                f"එය {province} පළාතේ ආරම්භ වී ගලා යයි / ගමන් කරයි. "
                f"ප්‍රදේශය අනුව ස්ථානීය නම් හා ලක්ෂණ වෙනස් විය හැකිය."
            )
        })
        qa_pairs.append({
            "instruction": f"Which districts does {name} pass through?",
            "input": "",
            "output": (
                f"{name} passes through the following districts: {spans_str_en}. "
                f"It spans multiple provinces in Sri Lanka. "
                f"Local names and characteristics may differ by region."
            )
        })
    
    # --- Core location Q&A (always generated, with full context) ---
    qa_pairs.append({
        "instruction": f"{name} කොහෙද?",
        "input": "",
        "output": (
            f"{name} ශ්‍රී ලංකාවේ {province} පළාතේ, {district} දිස්ත්‍රික්කයේ පිහිටා ඇත. "
            f"(GPS: {lat:.4f}°N, {lng:.4f}°E){disambig_suffix}"
        )
    })
    qa_pairs.append({
        "instruction": f"Where is {name} located in Sri Lanka?",
        "input": "",
        "output": (
            f"{name} is located in {district} District, {province} Province, Sri Lanka. "
            f"Coordinates: {lat:.4f}°N, {lng:.4f}°E.{disambig_suffix}"
        )
    })
    
    # --- Category & description Q&A ---
    if desc:
        qa_pairs.append({
            "instruction": f"{name} ගැන කියන්න.",
            "input": "",
            "output": f"{desc} ({district} දිස්ත්‍රික්කය, {province} පළාත, ශ්‍රී ලංකාව)"
        })
    
    # --- Best time / activities Q&A ---
    qa_pairs.append({
        "instruction": f"{name} නැරඹීමට හොඳම කාලය කවදාද?",
        "input": "",
        "output": f"{name} ({district}, {province}) නැරඹීමට හොඳම කාලය: {best_time}. ක්‍රියාකාරකම්: {activities}."
    })
    qa_pairs.append({
        "instruction": f"What is the best time to visit {name}?",
        "input": "",
        "output": f"The best time to visit {name} in {district}, {province} Province is: {best_time}. Popular activities: {activities}."
    })
    
    # --- Popularity / recommendation Q&A ---
    pop_map = {"High": "ඉතා ජනප්‍රිය", "Medium": "මධ්‍යම ජනප්‍රිය", "Low": "අඩු ජනප්‍රිය"}
    pop_si = pop_map.get(popularity, popularity)
    qa_pairs.append({
        "instruction": f"{name} ජනප්‍රිය ස්ථානයක්ද?",
        "input": "",
        "output": f"{name} ({district}, {province}) {pop_si} ශ්‍රී ලංකා සංචාරක ගමනාන්තයක් වේ."
    })
    
    return qa_pairs


def download_osm_pbf():
    """
    Downloads the Sri Lanka latest OSM PBF file from Geofabrik.
    """
    if os.path.exists(OSM_PBF_PATH):
        print(f"📂 Found existing OSM PBF file: {OSM_PBF_PATH}")
        return
        
    print(f"📥 Downloading Sri Lanka OSM PBF from {OSM_PBF_URL}...")
    start_time = time.time()
    
    def progress(count, block_size, total_size):
        elapsed = time.time() - start_time
        speed = (count * block_size) / (1024 * 1024 * elapsed) if elapsed > 0 else 0
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        print(f"\rDownloading: {percent}% completed | Speed: {speed:.2f} MB/s", end="", flush=True)

    urllib.request.urlretrieve(OSM_PBF_URL, OSM_PBF_PATH, reporthook=progress)
    print(f"\n✅ Download complete! File saved as {OSM_PBF_PATH}")

def main():
    global INPUT_FILE_PATH
    
    # --- PHASE 1: Extract/Parse Data ---
    if OSM_PBF_MODE:
        if osmiter is None:
            print("❌ Error: 'osmiter' is not installed. Cannot run in OSM PBF Mode.")
            print("Please run `pip install osmiter` first or disable OSM_PBF_MODE.")
            sys.exit(1)
            
        download_osm_pbf()
        
        print(f"🔍 Parsing OSM PBF file streamingly to extract up to {OSM_LIMIT_LOCATIONS} locations...")
        start_time = time.time()
        
        node_cache = {}
        extracted_locations = []
        count = 0
        
        try:
            # We iterate over all entities in the PBF file streamingly
            for feature in osmiter.iter_from_osm(OSM_PBF_PATH):
                if count >= OSM_LIMIT_LOCATIONS:
                    break
                    
                f_type = feature.get("type")
                if f_type == "node":
                    lat = feature.get("lat")
                    lng = feature.get("lon")
                    fid = feature.get("id")
                    if lat is not None and lng is not None and fid is not None:
                        node_cache[int(fid)] = (float(lat), float(lng))
                        
                        tags = feature.get("tag", {})
                        name = tags.get("name:en") or tags.get("name")
                        if name:
                            category = None
                            if "tourism" in tags:
                                category = tags["tourism"]
                            elif "historic" in tags:
                                category = tags["historic"]
                            elif "leisure" in tags:
                                category = tags["leisure"]
                            elif "natural" in tags:
                                category = tags["natural"]
                            elif "amenity" in tags and tags["amenity"] in ["place_of_worship", "restaurant", "cafe", "hotel", "scenic_viewpoint"]:
                                category = tags["amenity"]
                            elif "place" in tags and tags["place"] in ["town", "village", "island"]:
                                category = tags["place"]
                                
                            if category:
                                district, province = resolve_district_province(lat, lng)
                                extracted_locations.append({
                                    "name": str(name).strip(),
                                    "lat": lat,
                                    "lng": lng,
                                    "category": category.capitalize(),
                                    "district": district,
                                    "province": province,
                                    "opening_hours": tags.get("opening_hours", "Unknown"),
                                    "toilets": tags.get("toilets", "Unknown"),
                                    "wheelchair": tags.get("wheelchair", "Unknown"),
                                    "mobile_signal": tags.get("communication:mobile", "Unknown"),
                                    "Surfing": "yes" if "surfing" in tags or "surf" in name.lower() else "no",
                                    "Height_m": tags.get("height", "0"),
                                    "Length_km": "0"
                                })
                                count += 1
                                
                elif f_type == "way":
                    tags = feature.get("tag", {})
                    name = tags.get("name:en") or tags.get("name")
                    if name:
                        category = None
                        if "tourism" in tags:
                            category = tags["tourism"]
                        elif "historic" in tags:
                            category = tags["historic"]
                        elif "leisure" in tags:
                            category = tags["leisure"]
                        elif "natural" in tags:
                            category = tags["natural"]
                        elif "amenity" in tags and tags["amenity"] in ["place_of_worship", "restaurant", "cafe", "hotel", "scenic_viewpoint"]:
                            category = tags["amenity"]
                            
                        if category:
                            lats = []
                            lngs = []
                            for nid in feature.get("nd", []):
                                if nid in node_cache:
                                    lats.append(node_cache[nid][0])
                                    lngs.append(node_cache[nid][1])
                            if lats and lngs:
                                lat = sum(lats) / len(lats)
                                lng = sum(lngs) / len(lngs)
                                district, province = resolve_district_province(lat, lng)
                                extracted_locations.append({
                                    "name": str(name).strip(),
                                    "lat": lat,
                                    "lng": lng,
                                    "category": category.capitalize(),
                                    "district": district,
                                    "province": province,
                                    "opening_hours": tags.get("opening_hours", "Unknown"),
                                    "toilets": tags.get("toilets", "Unknown"),
                                    "wheelchair": tags.get("wheelchair", "Unknown"),
                                    "mobile_signal": tags.get("communication:mobile", "Unknown"),
                                    "Surfing": "yes" if "surfing" in tags or "surf" in name.lower() else "no",
                                    "Height_m": tags.get("height", "0"),
                                    "Length_km": "0"
                                })
                                count += 1
        except Exception as e:
            print(f"⚠️ Warning: Error while parsing PBF file: {e}")
            
        print(f"✅ Extracted {len(extracted_locations)} raw locations in {time.time() - start_time:.2f} seconds!")
        
        # Create a DataFrame
        df = pd.DataFrame(extracted_locations)
        # Save raw extraction for references
        raw_csv_path = "sri_lanka_extracted_locations.csv"
        df.to_csv(raw_csv_path, index=False, encoding="utf-8")
        print(f"📂 Extracted raw locations saved to: {raw_csv_path}")
        
    else:
        # Load CSV or Excel File Mode
        if INPUT_FILE_PATH and os.path.exists(INPUT_FILE_PATH):
            csv_file_path = INPUT_FILE_PATH
            print(f"📂 Using pre-configured file: {csv_file_path}")
        else:
            try:
                from google.colab import files
                print("📂 Please select and upload your CSV or Excel file:")
                uploaded = files.upload()
                if not uploaded:
                    print("❌ No file uploaded. Exiting.")
                    sys.exit(1)
                csv_file_path = list(uploaded.keys())[0]
            except ImportError:
                csv_file_path = input("💻 Enter local CSV or Excel file path: ").strip()
                
        if not os.path.exists(csv_file_path):
            print(f"❌ File {csv_file_path} not found.")
            return

        file_ext = os.path.splitext(csv_file_path)[1].lower()
        if file_ext in [".xlsx", ".xls"]:
            print(f"\n📂 Reading Excel File: {csv_file_path}...")
            try:
                all_sheets = pd.read_excel(csv_file_path, sheet_name=None)
                df_list = []
                for sheet_name, sheet_df in all_sheets.items():
                    clean_sheet_df = sheet_df.dropna(how='all')
                    clean_sheet_df["__sheet_name__"] = sheet_name
                    df_list.append(clean_sheet_df)
                df = pd.concat(df_list, ignore_index=True)
            except Exception as e:
                print(f"❌ Error reading Excel: {e}")
                return
        else:
            print(f"\n📂 Reading CSV File: {csv_file_path}...")
            try:
                df = pd.read_csv(csv_file_path)
            except Exception:
                df = pd.read_csv(csv_file_path, encoding="latin1")

    # --- PHASE 2: Setup processing configuration ---
    # Setup LLM if we are not in fast heuristic mode
    if not OSM_FAST_HEURISTIC_MODE:
        tokenizer, model, device = setup_model()

    final_dataset = []
    completed_names = set()
    
    # Recovery Mode (load existing progress backup)
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                for p in loaded_data:
                    name = p.get("name")
                    if name:
                        completed_names.add(name)
                        final_dataset.append(p)
            print(f"🔄 Recovered {len(completed_names)} records from previous backup.")
        except Exception:
            pass

    # --- PHASE 3: Processing and Compiling ---
    loop_start_time = time.time()
    print(f"🚀 Compiling metadata for {len(df)} locations...")
    
    for index, row in df.iterrows():
        # Check time limit
        elapsed_time = time.time() - loop_start_time
        if elapsed_time > MAX_RUN_TIME_SECONDS:
            print(f"\n⚠️ Time limit reached. Saving progress and stopping.")
            break

        place_name = row.get("name", "")
        if not place_name or pd.isna(place_name):
            continue
            
        if place_name in completed_names:
            continue
            
        lat = float(row.get("lat", 0.0))
        lng = float(row.get("lng", 0.0))
        category = str(row.get("category", "Scenic Spot"))
        district = str(row.get("district", "Unknown"))
        province = str(row.get("province", "Unknown"))
        opening_hours_osm = str(row.get("opening_hours", "Unknown"))

        if OSM_FAST_HEURISTIC_MODE:
            # High-speed rule-based compilation (takes microseconds)
            tripme_data = get_heuristic_metadata(place_name, category, district, province, lat, lng, opening_hours_osm)
            tripme_data["id"] = f"pl_{hashlib.md5(place_name.encode('utf-8')).hexdigest()[:8]}"
            tripme_data["name"] = place_name
            tripme_data["district_id"] = district
            tripme_data["province_id"] = province
            tripme_data["category_id"] = category
            tripme_data["family_friendly"] = "yes"
            tripme_data["Surfing"] = str(row.get("Surfing", "no"))
            
        else:
            # LLM-based compilation (calls AI for each record)
            print(f"⚡ LLM Processing row {index+1}/{len(df)}: {place_name}...", end=" ", flush=True)
            existing_dict = row.to_dict()
            ai_data = generate_and_verify(tokenizer, model, device, place_name, existing_dict)
            
            if not ai_data:
                print("❌ Failed (using fallback rules)")
                tripme_data = get_heuristic_metadata(place_name, category, district, province, lat, lng, opening_hours_osm)
            else:
                print("✅ Success!")
                tripme_data = {
                    "id": f"pl_{hashlib.md5(place_name.encode('utf-8')).hexdigest()[:8]}",
                    "name": place_name,
                    "description": ai_data.get("description", "A beautiful tourist place located in Sri Lanka."),
                    "district_id": district,
                    "province_id": province,
                    "category_id": category,
                    "lat": float(ai_data.get("latitude", lat)),
                    "lng": float(ai_data.get("longitude", lng)),
                    "opening_hours": ai_data.get("opening_hours", opening_hours_osm),
                    "mobile_signal": ai_data.get("mobile_signal", "Good"),
                    "road_condition": ai_data.get("road_condition", "Paved"),
                    "activities": ", ".join(ai_data.get("activities", [])) if isinstance(ai_data.get("activities"), list) else str(ai_data.get("activities", "Sightseeing")),
                    "tourist_popularity": ai_data.get("tourist_popularity", "Medium"),
                    "family_friendly": "yes",
                    "budget_category": ai_data.get("budget_category", "Free"),
                    "ticket_price": ai_data.get("ticket_price", "Free"),
                    "parking_avail": "yes" if "yes" in str(ai_data.get("parking_avail", "yes")).lower() else "no",
                    "toilets": "yes" if "yes" in str(ai_data.get("toilets", "yes")).lower() else "no",
                    "food_nearby": "yes" if "yes" in str(ai_data.get("food_nearby", "yes")).lower() else "no",
                    "wheelchair_access": "yes" if "yes" in str(ai_data.get("wheelchair_access", "no")).lower() else "no",
                    "camping_allowed": "yes" if "yes" in str(ai_data.get("camping_allowed", "no")).lower() else "no",
                    "safety_level": ai_data.get("safety_level", "Safe"),
                    "wildlife_hazard": ai_data.get("wildlife_hazard", "None"),
                    "guide_required": "yes" if "yes" in str(ai_data.get("guide_required", "no")).lower() else "no",
                    "rain_sensitivity": ai_data.get("rain_sensitivity", "Safe"),
                    "monsoon_note": ai_data.get("monsoon_note", "None"),
                    "best_time_to_visit": ai_data.get("best_time_to_visit", "All year round"),
                    "Height_m": str(ai_data.get("height_m", "0")),
                    "Length_km": str(ai_data.get("length_km", "0")),
                    "Surfing": str(row.get("Surfing", "no"))
                }
        
        final_dataset.append(tripme_data)
        completed_names.add(place_name)
        
        # Save progress every 50 records in fast mode, or 5 in LLM mode
        save_interval = 250 if OSM_FAST_HEURISTIC_MODE else 5
        if len(final_dataset) % save_interval == 0:
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    # --- PHASE 4: Save & Export ---
    # 4a. Save full JSON output
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 Compilation complete! {len(final_dataset)} records successfully processed!")
    print(f"📄 JSON saved to: {OUTPUT_JSON_PATH}")
    
    # 4b. CSV Export with fixed, consistent column order
    CSV_COLUMNS = [
        "id", "name", "category_id", "district_id", "province_id",
        "lat", "lng", "description", "best_time_to_visit", "ticket_price",
        "opening_hours", "activities", "tourist_popularity", "budget_category",
        "parking_avail", "toilets", "food_nearby", "wheelchair_access",
        "camping_allowed", "family_friendly", "safety_level", "wildlife_hazard",
        "guide_required", "rain_sensitivity", "monsoon_note", "mobile_signal",
        "road_condition", "Height_m", "Length_km", "Surfing"
    ]
    try:
        final_df = pd.DataFrame(final_dataset)
        for col in CSV_COLUMNS:
            if col not in final_df.columns:
                final_df[col] = ""
        final_df = final_df[CSV_COLUMNS]
        # UTF-8 BOM so Excel opens Sinhala/Unicode text correctly
        final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"📊 CSV database exported to: {OUTPUT_CSV_PATH}")
        print(f"   Columns: {len(CSV_COLUMNS)} | Rows: {len(final_df)}")
    except Exception as e:
        print(f"⚠️ Failed to export to CSV: {e}")
    
    # 4c. SFT Training JSONL Export (disambiguation-safe Q&A pairs)
    # Generates Alpaca-format training pairs with full location context embedded.
    # Each record produces 5-7 Q&A pairs. This prevents place-name confusion during training.
    print(f"\n🧠 Generating disambiguation-safe SFT training pairs...")
    sft_records = []
    for record in final_dataset:
        qa_pairs = generate_disambiguation_qa(record)
        for qa in qa_pairs:
            sft_records.append(qa)
    
    try:
        with open(OUTPUT_SFT_PATH, "w", encoding="utf-8") as f:
            for rec in sft_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"✅ SFT training JSONL saved to: {OUTPUT_SFT_PATH}")
        print(f"   Total Q&A pairs: {len(sft_records)} (from {len(final_dataset)} locations)")
    except Exception as e:
        print(f"⚠️ Failed to export SFT JSONL: {e}")

    # Trigger automatic browser download if in Colab
    try:
        from google.colab import files
        print(f"\n📥 Downloading files to your local computer...")
        files.download(OUTPUT_JSON_PATH)
        if os.path.exists(OUTPUT_CSV_PATH):
            files.download(OUTPUT_CSV_PATH)
        if os.path.exists(OUTPUT_SFT_PATH):
            files.download(OUTPUT_SFT_PATH)
        print("✅ Download prompt triggered successfully!")
    except ImportError:
        print(f"💻 Local environment. Files saved at:\n   JSON: {os.path.abspath(OUTPUT_JSON_PATH)}\n   CSV:  {os.path.abspath(OUTPUT_CSV_PATH)}\n   SFT:  {os.path.abspath(OUTPUT_SFT_PATH)}")


if __name__ == "__main__":
    main()
