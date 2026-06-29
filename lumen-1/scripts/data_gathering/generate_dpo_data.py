import json
import os
import random

# Input/Output Paths
OSM_INPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sft_osm_data.jsonl"))
OUTPUT_DPO_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dpo_data.jsonl"))

# Sinhala translation dictionary for common travel terms
SINHALA_DICTIONARY = {
    "beaches": "වෙරළ තීරයක්",
    "beach": "වෙරළක්",
    "waterfalls": "දියඇල්ලක්",
    "waterfall": "දියඇල්ලක්",
    "mountains": "කඳු වැටියක්",
    "mountain": "කන්දක්",
    "temples": "ඓතිහාසික විහාරස්ථානයක්",
    "temple": "විහාරස්ථානයක්",
    "parks": "ජාතික උද්‍යානයක්",
    "park": "උද්‍යානයක්",
    "lakes": "වැවක්/ජලාශයක්",
    "lake": "වැවක්",
    "safe": "සම්පූර්ණයෙන්ම ආරක්ෂිතයි",
    "moderate": "සාමාන්‍ය ආරක්ෂිතයි",
    "dangerous": "අවදානම් සහිතයි",
    "yes": "පහසුකම් ඇත",
    "no": "පහසුකම් නොමැත",
    "free": "නොමිලේ",
}

def clean_desc(desc):
    if not desc or "beautiful beaches" in desc.lower() or "no description" in desc.lower():
        return ""
    return desc

def get_sinhala_term(term):
    if not term:
        return "නොදනී"
    return SINHALA_DICTIONARY.get(str(term).strip().lower(), term)

def generate_sinhala_description(name, category, desc, safety, parking, toilets):
    category_sin = get_sinhala_term(category)
    safety_sin = get_sinhala_term(safety)
    parking_sin = "වාහන නැවැත්වීමේ පහසුකම් ඇත" if str(parking).lower() == "yes" else "වාහන නැවැත්වීමට නිශ්චිත ඉඩක් නොමැත"
    toilets_sin = "වැසිකිළි පහසුකම් ඇත" if str(toilets).lower() == "yes" else "වැසිකිළි පහසුකම් නොමැත"
    
    base = f"{name} යනු ශ්‍රී ලංකාවේ පිහිටි ඉතා සුන්දර {category_sin} වේ."
    if desc:
        desc_sin = desc.replace("beautiful", "ලස්සන").replace("beach", "වෙරළ").replace("waterfall", "දියඇල්ල")
        details = f" මෙම ස්ථානය පිළිබඳව: {desc_sin}"
    else:
        details = " මෙය සොබාදහමේ සුන්දරත්වය විඳගත හැකි අපූරු ස්ථානයකි."
        
    safety_desc = f" මෙම ප්‍රදේශය සංචාරකයින් සඳහා {safety_sin} මට්ටමක පවතී."
    facilities = f" මෙහි {parking_sin} සහ {toilets_sin}."
    return f"{base}{details}{safety_desc}{facilities}"

def main():
    print("🚀 Generating DPO chosen/rejected dataset...")
    
    if not os.path.exists(OSM_INPUT_FILE):
        print(f"❌ Input file {OSM_INPUT_FILE} not found!")
        return
        
    dpo_dataset = []
    
    # 1. Add specific high-priority geographic disambiguation pairs (e.g. Nuwara Wewa)
    custom_pairs = [
        {
            "prompt": "Where is Nuwara Wewa located?",
            "chosen": "Nuwara Wewa is located in the Anuradhapura district, North Central Province of Sri Lanka. Despite the word 'Nuwara' in its name, it is situated in Anuradhapura, not Kandy or Nuwara Eliya.",
            "rejected": "Nuwara Wewa is located in the Nuwara Eliya district or Kandy district of Sri Lanka."
        },
        {
            "prompt": "නුවර වැව පිහිටා තිබෙන්නේ කොහේද?",
            "chosen": "නුවර වැව පිහිටා තිබෙන්නේ අනුරාධපුර දිස්ත්‍රික්කයේ ය. එහි නමට 'නුවර' යන වචනය තිබුණද, එය මහනුවර (Kandy) හෝ නුවරඑළිය දිස්ත්‍රික්කවලට අයත් නොවේ.",
            "rejected": "නුවර වැව පිහිටා තිබෙන්නේ මහනුවර හෝ නුවරඑළිය දිස්ත්‍රික්කයේ ය."
        },
        {
            "prompt": "Tell me about Nuwara Wewa.",
            "chosen": "Nuwara Wewa is a historic reservoir (wewa) located in Anuradhapura, Sri Lanka. It was built in the 1st century BC. Despite the name 'Nuwara', it is not in Nuwara Eliya or Kandy; it is situated in the North Central Province.",
            "rejected": "Nuwara Wewa is a lake located in Nuwara Eliya where tourists can enjoy boating and cool climate."
        }
    ]
    dpo_dataset.extend(custom_pairs)
    
    # 2. Extract and compile from OSM data
    with open(OSM_INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                messages = record.get("messages", [])
                if len(messages) < 2:
                    continue
                
                assistant_content_str = messages[1].get("content", "{}")
                place_details = json.loads(assistant_content_str)
                
                place_name = place_details.get("name", "Unknown Place")
                desc = clean_desc(place_details.get("description", ""))
                category = place_details.get("category_id", "place")
                parking = place_details.get("parking_avail", "no")
                toilets = place_details.get("toilets", "no")
                safety = place_details.get("safety_level", "Safe")
                
                # English Pair
                eng_prompt = f"Describe {place_name} in Sri Lanka."
                eng_chosen = f"{place_name} is a popular {category} in Sri Lanka."
                if desc:
                    eng_chosen += f" {desc}"
                else:
                    eng_chosen += f" It offers scenic beauty, ideal for relaxing and photography."
                eng_chosen += f"\n- Safety Level: {safety}\n- Parking: {parking} | Toilets: {toilets}"
                
                eng_rejected = f"This is a place called {place_name} in Sri Lanka. No specific description or details are available."
                
                dpo_dataset.append({
                    "prompt": eng_prompt,
                    "chosen": eng_chosen,
                    "rejected": eng_rejected
                })
                
                # Sinhala Pair
                sin_prompt = f"ලංකාවේ {place_name} ගැන විස්තරයක් කියන්න."
                sin_chosen = generate_sinhala_description(place_name, category, desc, safety, parking, toilets)
                sin_rejected = f"මෙය ශ්‍රී ලංකාවේ පිහිටි {place_name} නම් ස්ථානයකි. වැඩි විස්තර නොමැත."
                
                dpo_dataset.append({
                    "prompt": sin_prompt,
                    "chosen": sin_chosen,
                    "rejected": sin_rejected
                })
                
            except Exception:
                pass
                
    # Shuffle for training diversity
    random.shuffle(dpo_dataset)
    
    # Save to DPO JSONL format
    os.makedirs(os.path.dirname(OUTPUT_DPO_FILE), exist_ok=True)
    with open(OUTPUT_DPO_FILE, 'w', encoding='utf-8') as f:
        for item in dpo_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ Generated {len(dpo_dataset)} preference pairs in {OUTPUT_DPO_FILE}!")

if __name__ == "__main__":
    main()
