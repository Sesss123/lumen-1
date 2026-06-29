import pandas as pd
import torch
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Configuration ---
EXCEL_FILE_PATH = "places.xlsx" 
INPUT_JSON_PATH = "tripme_database_complete.json"     # ඔයා කලින් හදලා Upload කරපු ෆයිල් එක
OUTPUT_JSON_PATH = "tripme_database_complete_NEW.json" # අලුතින් හැදෙන ෆයිල් එක
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 

def setup_model():
    print("🚀 Loading Qwen 2.5 model in pure float16 mode for ULTRA SPEED...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def generate_and_verify(tokenizer, model, place_name, existing_data_dict):
    existing_data_str = json.dumps(existing_data_dict, ensure_ascii=False)
    
    prompt = f"""You are an expert Sri Lankan travel guide and data compiler.
Place: {place_name}
Existing Data from CSV: {existing_data_str}

Task: You must generate a complete JSON profile for this place. 
Use your knowledge to fill in EVERY field with realistic and accurate details. Do NOT just output "no" or "unknown" if you can make a highly educated guess. 
For example, for a popular beach, parking and toilets are usually "yes". Ticket price is usually "Free". 

CRITICAL: The 'description' MUST be highly specific to '{place_name}'. Do NOT write a generic description. It must be an engaging, 50-word educational description highlighting the unique features, history, or beauty of THIS EXACT PLACE.

Return strictly ONLY a valid JSON object matching this exact format:
{{
    "description": "Write the highly specific 50-word rich description here",
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

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # වැඩිපුර දත්ත ඉල්ලන නිසා Tokens ගාණ 800 ක් කළා. නැත්නම් මැදින් කැපිලා ගිහින් Error එනවා.
    outputs = model.generate(**inputs, max_new_tokens=800, temperature=0.2)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        return None
    except Exception:
        return None

def extract_basic_info(row):
    name = ""
    for col in ["Beach_Name", "River_Name", "Waterfall_Name", "Lake_Name", "Viewpoint_Name", "National_Park", "Name", "Place_Name"]:
        if col in row and pd.notna(row[col]):
            name = str(row[col]).strip()
            break
            
    # අකුරු සහ ඉලක්කම් කලවම් වුණු Unique ID එකක් හදනවා (MD5 Hash එකක් පාවිච්චි කරලා)
    # මේකෙන් වාසියක් තියෙනවා: එකම නමට හැමදාම හැදෙන්නේ එකම ID එකයි (උදා: pl_7a3b8c2d)
    import hashlib
    if name:
        hash_id = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        place_id = f"pl_{hash_id}"
    else:
        import uuid
        place_id = f"pl_{uuid.uuid4().hex[:8]}"
            
    existing_data = {}
    for col, val in row.items():
        if pd.notna(val) and str(val).strip() != "":
            existing_data[col] = val
            
    return place_id, name, existing_data

def safe_float(val):
    try:
        if val is None or str(val).strip() == "" or str(val).lower() == "no":
            return 0.0
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
        if numbers:
            return float(numbers[0])
        return 0.0
    except Exception:
        return 0.0

def main():
    tokenizer, model = setup_model()
    
    print(f"\n📂 Loading Excel file: {EXCEL_FILE_PATH}...")
    try:
        all_sheets = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
    except FileNotFoundError:
        print(f"❌ '{EXCEL_FILE_PATH}' හොයාගන්න බෑ.")
        return

    final_dataset = []
    completed_names = set()
    
    if os.path.exists(INPUT_JSON_PATH):
        try:
            with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                
                # කලින් හදපු ඒවගෙන්, හරිගිය ඒවා විතරක් වෙන් කරගන්නවා
                for p in loaded_data:
                    desc = str(p.get("description", "")).strip().lower()
                    name = p.get("name")
                    
                    # අලුත් Format එකේ තියෙන දේවල් (id, opening_hours) තියෙනවද කියලත් බලනවා
                    has_new_format = "opening_hours" in p and "id" in p
                    
                    if name and desc != "no" and desc != "" and len(desc) > 10 and has_new_format:
                        # කලින් හැදිලා තියෙන එකේ ID එක පරණ විදියට (නම විදියට) තියෙනවා නම්,
                        # ඒක මෙතනදිම අලුත් ඉලක්කම් ID එකට Update කරනවා (AI එක ආයේ Run කරන්නේ නැතුව)
                        import hashlib
                        hash_id = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
                        p["id"] = f"pl_{hash_id}"
                        
                        completed_names.add(name)
                        final_dataset.append(p) 
                        
            print(f"🔄 කලින් 100% සාර්ථකව (අලුත් Format එකෙන්) හැදුණු ස්ථාන {len(completed_names)} ක් හම්බුණා. ඒවා ආයේ හදන්නේ නෑ.")
            print(f"⚠️ කලින් හැදුණු හැබැයි පරණ Format එකේ තියෙන ඒවා සහ 'no' වැටිලා අසාර්ථක වුණු ඒවා ආයෙත් අලුතින් හදනවා.")
        except Exception:
            pass

    for sheet_name, df in all_sheets.items():
        print(f"\n📑 Processing Sheet: {sheet_name}")
        
        for index, row in df.iterrows():
            place_id, place_name, existing_data = extract_basic_info(row)
            
            if not place_name or place_name in completed_names:
                continue 
                
            print(f"⚡ Generating data for: {place_name}...", end=" ", flush=True)
            ai_data = generate_and_verify(tokenizer, model, place_name, existing_data)
            
            if not ai_data:
                ai_data = {} 
                print("❌ Failed (JSON Error)")
            else:
                print("✅ Success!")

            category = str(existing_data.get("Category", existing_data.get("Type", "Unknown")))
            district = str(existing_data.get("District", "Unknown"))
            province = str(existing_data.get("Province", "Unknown"))
            
            # AI එකෙන් හරි, Excel එකෙන් හරි GPS අගයන් හොයාගන්නවා
            lat = safe_float(ai_data.get("latitude", 0.0))
            if lat == 0.0:
                lat = safe_float(existing_data.get("Latitude", 0.0))
                
            lng = safe_float(ai_data.get("longitude", 0.0))
            if lng == 0.0:
                lng = safe_float(existing_data.get("Longitude", 0.0))

            tripme_data = {
                "id": place_id,
                "name": place_name,
                "description": ai_data.get("description", "A beautiful place located in Sri Lanka."),
                "district_id": district,
                "province_id": province,
                "category_id": category,
                "lat": lat,
                "lng": lng,
                "opening_hours": ai_data.get("opening_hours", "Unknown"),
                "mobile_signal": ai_data.get("mobile_signal", "Unknown"),
                "road_condition": ai_data.get("road_condition", "Unknown"),
                "activities": ai_data.get("activities", "Sightseeing"),
                "tourist_popularity": ai_data.get("tourist_popularity", str(existing_data.get("Tourist_Popularity", "High"))),
                "family_friendly": ai_data.get("family_friendly", "yes"),
                "budget_category": ai_data.get("budget_category", "Free"),
                "ticket_price": ai_data.get("ticket_price", "Free"),
                "parking_avail": ai_data.get("parking_avail", "yes"),
                "toilets": ai_data.get("toilets", "yes"),
                "food_nearby": ai_data.get("food_nearby", "yes"),
                "wheelchair_access": ai_data.get("wheelchair_access", "no"),
                "camping_allowed": ai_data.get("camping_allowed", "no"),
                "safety_level": ai_data.get("safety_level", "Safe"),
                "wildlife_hazard": ai_data.get("wildlife_hazard", "None"),
                "guide_required": ai_data.get("guide_required", "no"),
                "rain_sensitivity": ai_data.get("rain_sensitivity", "Safe"),
                "monsoon_note": ai_data.get("monsoon_note", "None"),
                "best_time_to_visit": ai_data.get("best_time_to_visit", "All year round"),
                
                # Height/Length - AI එකෙන් හරි, Excel එකෙන් හරි ගන්නවා
                "Height_m": str(ai_data.get("height_m", existing_data.get("Height_m", "0"))),
                "Length_km": str(ai_data.get("length_km", existing_data.get("Length_km", "0"))),
                "Surfing": str(existing_data.get("Surfing", "no"))
            }
            
            final_dataset.append(tripme_data)
            completed_names.add(place_name)
            
            if len(final_dataset) % 5 == 0:
                with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(final_dataset, f, ensure_ascii=False, indent=4)
                    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 සම්පූර්ණයෙන්ම ඉවරයි! {len(final_dataset)} ක් දත්ත {OUTPUT_JSON_PATH} විදියට Save වුණා!")

if __name__ == "__main__":
    main()
