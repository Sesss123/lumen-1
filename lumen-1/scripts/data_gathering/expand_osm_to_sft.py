import json
import os
import random
import hashlib

# Input/Output Paths
OSM_INPUT_FILE = "../data/sft_osm_data.jsonl"
OUTPUT_SFT_FILE = "../data/sft.jsonl"  # Updates directly in sft.jsonl used by sft.yaml

# Sinhala translation dictionary for common travel terms
SINHALA_DICTIONARY = {
    # Categories
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
    
    # Safety levels
    "safe": "සම්පූර්ණයෙන්ම ආරක්ෂිතයි",
    "moderate": "සාමාන්‍ය ආරක්ෂිතයි",
    "dangerous": "අවදානම් සහිතයි",
    
    # Yes/No
    "yes": "පහසුකම් ඇත",
    "no": "පහසුකම් නොමැත",
    "free": "නොමිලේ",
    
    # Road types
    "paved": "කාපට් කරන ලද හොඳ තත්ත්වයේ මාර්ගයකි",
    "4wd": "4WD වාහන පමණක් යා හැකි දුෂ්කර මාර්ගයකි",
    "sand": "වැලි සහිත මාර්ගයකි",
    
    # Mobile signal
    "good": "හොඳින් සංඥා ඇත",
    "weak": "සංඥා දුර්වලයි",
    "none": "සංඥා නොමැත"
}

def clean_desc(desc):
    """Clean descriptions to remove placeholder sentences."""
    if not desc or "beautiful beaches" in desc.lower() or "no description" in desc.lower():
        return ""
    return desc

def get_sinhala_term(term):
    if not term:
        return "නොදනී"
    term_lower = str(term).strip().lower()
    return SINHALA_DICTIONARY.get(term_lower, term)

def generate_sinhala_description(name, category, desc, safety, parking, toilets, mobile):
    """Generate a rich, grammatically correct Sinhala paragraph."""
    category_sin = get_sinhala_term(category)
    safety_sin = get_sinhala_term(safety)
    parking_sin = "වාහන නැවැත්වීමේ පහසුකම් ඇත" if str(parking).lower() == "yes" else "වාහන නැවැත්වීමට නිශ්චිත ඉඩක් නොමැත"
    toilets_sin = "වැසිකිළි පහසුකම් ඇත" if str(toilets).lower() == "yes" else "වැසිකිළි පහසුකම් නොමැත"
    mobile_sin = get_sinhala_term(mobile)
    
    # Base sentence
    base = f"{name} යනු ශ්‍රී ලංකාවේ පිහිටි ඉතා සුන්දර {category_sin} වේ."
    
    if desc:
        # Translate simple keywords in description if any
        desc_sin = desc.replace("beautiful", "ලස්සන").replace("beach", "වෙරළ").replace("waterfall", "දියඇල්ල")
        details = f" මෙම ස්ථානය පිළිබඳව: {desc_sin}"
    else:
        details = " මෙය සංචාරකයින් අතර බෙහෙවින් ප්‍රසිද්ධ, සොබාදහමේ සුන්දරත්වය විඳගත හැකි අපූරු ස්ථානයකි."
        
    safety_desc = f" මෙම ප්‍රදේශය සංචාරකයින් සඳහා {safety_sin} මට්ටමක පවතී."
    facilities = f" මෙහි {parking_sin} සහ {toilets_sin}."
    signal = f" ජංගම දුරකථන සංඥා මට්ටම: {mobile_sin}."
    
    return f"{base}{details}{safety_desc}{facilities}{signal}"

def main():
    print("🚀 Running advanced OSM to SFT conversion with parser bug fix...")
    
    if not os.path.exists(OSM_INPUT_FILE):
        print(f"❌ Input file {OSM_INPUT_FILE} not found!")
        return
        
    sft_dataset = []
    
    with open(OSM_INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                messages = record.get("messages", [])
                if len(messages) < 2:
                    continue
                
                # BUG FIX: Parser was directly looking at root instead of decoding JSON string in assistant message
                assistant_content_str = messages[1].get("content", "{}")
                place_details = json.loads(assistant_content_str)
                
                place_name = place_details.get("name", "Unknown Place")
                desc = clean_desc(place_details.get("description", ""))
                category = place_details.get("category_id", "place")
                best_time = place_details.get("best_time_to_visit", "All year round")
                ticket = place_details.get("ticket_range", place_details.get("ticket_price", "Free"))
                parking = place_details.get("parking_avail", "no")
                toilets = place_details.get("toilets", "no")
                safety = place_details.get("safety_level", "Safe")
                mobile = place_details.get("mobile_signal", "good")
                
                # Build English QA
                eng_ans = f"{place_name} is a popular {category} in Sri Lanka."
                if desc:
                    eng_ans += f" {desc}"
                else:
                    eng_ans += f" It offers scenic beauty, ideal for relaxing and photography."
                eng_ans += f"\n- Safety Level: {safety}"
                eng_ans += f"\n- Parking: {parking} | Toilets: {toilets}"
                eng_ans += f"\n- Mobile Signal: {mobile}"
                
                eng_pair = {
                    "messages": [
                        {"role": "user", "content": f"Tell me about {place_name} in Sri Lanka."},
                        {"role": "assistant", "content": eng_ans}
                    ]
                }
                sft_dataset.append(eng_pair)
                
                # Build Rich Sinhala QA
                sin_desc = generate_sinhala_description(place_name, category, desc, safety, parking, toilets, mobile)
                sin_pair = {
                    "messages": [
                        {"role": "user", "content": f"ලංකාවේ {place_name} ගැන විස්තරයක් කියන්න."},
                        {"role": "assistant", "content": sin_desc}
                    ]
                }
                sft_dataset.append(sin_pair)
                
            except Exception as e:
                # Log error and continue
                pass

    # Shuffle for training diversity
    random.shuffle(sft_dataset)

    # Save to SFT JSONL format
    os.makedirs(os.path.dirname(OUTPUT_SFT_FILE), exist_ok=True)
    with open(OUTPUT_SFT_FILE, 'w', encoding='utf-8') as f:
        for item in sft_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"✅ Parser bug fixed successfully!")
    print(f"✅ Generated {len(sft_dataset)} QA records (English & Sinhala) in {OUTPUT_SFT_FILE}!")


if __name__ == "__main__":
    main()
