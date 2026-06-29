# =====================================================================
# AI Nutrition Database Generator (Google Colab & Local GPU Compatible)
# =====================================================================
# මෙම ස්ක්‍රිප්ට් එක Google Colab තුළ හෝ Local GPU එකක Run කර Qwen/Mistral
# AI Model එකක් හරහා ශ්‍රී ලාංකික හා ආසියානු ආහාර/බීම වර්ග 1,500ක
# නිශ්චිත පෝෂණ ගුණ (Calories, Protein, Carbs, Fat) ස්වයංක්‍රීයව Generate කරයි.
# =====================================================================

import os
import json
import time
import re
from typing import Dict, Any, List

# Colab එක තුළ ක්‍රියාත්මක වන්නේ නම් Auto-Download සඳහා files import කරගැනීම
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Colab එක තුළ අවශ්‍ය packages නොමැති නම් දැනුම් දීම
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# =====================================================================
# තෝරාගත හැකි AI Models (Target Open-Source LLMs for Download)
# =====================================================================
# මෙම ස්ක්‍රිප්ට් එක Run වන විට HuggingFace වෙතින් පහත AI Model එක Download වේ:
# 1. "Qwen/Qwen2.5-1.5B-Instruct" (Selected for 5x Speed) - Colab Free Tier මත
#    ඉතා වේගයෙන් Download (3GB) වී ක්ෂණිකව ආහාර උත්පාදනය කරන මාදිලියකි.
# 2. "Qwen/Qwen2.5-7B-Instruct" - වඩාත් විශාල හා ගුණාත්මක විකල්ප මාදිලිය (15GB).
# 3. "mistralai/Mistral-7B-Instruct-v0.3" - විකල්ප වේගවත් Model එකකි.
# =====================================================================
AI_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Colab Timeout වීම වැළැක්වීම සඳහා උපරිම කාල සීමාව (පැය 5 = තත්පර 18,000)
MAX_EXECUTION_TIME_SECONDS = 5 * 3600
OUTPUT_FILE_PATH = "sri_lankan_nutrition_2500.json"

# ආහාර උත්පාදනය කළ යුතු ප්‍රධාන කාණ්ඩ 21 (21 Food Categories for 2500 items)
CATEGORIES = [
    "Traditional Sri Lankan Rice Varieties & Rice Combos (e.g., Yellow Rice, Lumprais, Ghee Rice)",
    "Authentic Curries & Gravies (e.g., Black Pork Curry, Cuttlefish Curry, Crab Curry)",
    "Sambols, Salads & Mallum (e.g., Kathurumurunga Mallum, Mukunuwenna, Pennywort)",
    "Hoppers, Roti & Pittu Combos (e.g., Egg Roti, Godamba Roti, Roast Bread with Dhal)",
    "Street Foods & Kottu Varieties (e.g., Dolphin Kottu, String Hopper Kottu, Roast Chicken Kottu)",
    "Traditional Sweets & Kavum (e.g., Kokis, Mung Kavum, Asmi, Kalu Dodol, Thalaguli)",
    "Beverages, Teas & Fresh Juices (e.g., King Coconut Water, Woodapple Juice, Passion Fruit Juice, Plain Tea with Sugar)",
    "Herbal Porridge / Kola Kenda (e.g., Hathawariya Kenda, Karapincha Kenda, Welpenela Kenda)",
    "Bakery Items & Sri Lankan Short Eats (e.g., Fish Bun / Maalu Paan, Seeni Sambol Bun, Egg Bun, Chinese Roll)",
    "South Asian Dosa & Idli Combos (e.g., Masala Dosa, Ghee Roast Dosa, Medu Vada)",
    "Fried Foods & Bites (e.g., Fried Cassava Chips, Fried Breadfruit, Fried Sprats / Halmasso)",
    "Seafood Specialties (e.g., Grilled Jumbo Prawns, Devilled Cuttlefish, Hot & Sweet Fish)",
    "Meat & Poultry Dishes (e.g., Devilled Chicken, Pork Mustard, Beef Stew, Mutton Curry)",
    "Vegetarian & Vegan Specialties (e.g., Tempered Chickpeas / Kadala, Green Gram / Mung Kiribath)",
    "Daily Fruits & Snacks (e.g., Rambutan, Mangosteen, Durian, Red Banana, Anoda / Soursop)",
    "Traditional Ayurvedic & Herbal Beverages (e.g., Paspanguwa, Samahan, Ranawara, Belimal, Arishta varieties)",
    "New Year Festival Sweets & Treats (e.g., Asmi, Aluva, Wel Thalapa, Mun Kavum, Kalu Dodol)",
    "Modern Cafe & Fast Food Items in Sri Lanka (e.g., Chicken Submarine, Cheese Burger, Chicken Pizza, Boba Milk Tea, Shawarma)",
    "Regional Sri Lankan Specialties (e.g., Jaffna Crab Curry, Odiyal Kool, Batticaloa Mutton Curry, Southern Fish Ambul Thiyal)",
    "Commercial Biscuits & Packaged Snacks (e.g., Marie Biscuit, Lemon Puff, Ginger Biscuit, Tipi Tip, Cream Cracker)",
    "Nutritional Supplements & Malt Drinks (e.g., Thriposha, Samaposha, Nestomalt, Milo, Horlicks)"
]

def save_json_incrementally(data: Dict[str, Dict[str, Any]], filepath: str):
    """
    සෑම කාණ්ඩයක් අවසානයේම දත්ත ගොනුවට ස්වයංක්‍රීයව Save කරන ශ්‍රිතය.
    (Colab Disconnect වූවත් දත්ත ආරක්ෂා වේ).
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def trigger_colab_auto_download(filepath: str):
    """
    Colab එකේ Run වන්නේ නම් ගොනුව ස්වයංක්‍රීයව Laptop එකට Download කරයි.
    """
    if IN_COLAB and os.path.exists(filepath):
        print(f"\n📥 [Auto-Download] '{filepath}' ගොනුව ඔබගේ Laptop එකට Download වෙමින් පවතී...")
        try:
            files.download(filepath)
            print("✅ Download කිරීම සාර්ථකව ආරම්භ විය!")
        except Exception as e:
            print(f"⚠️ Auto-Download දෝෂයක්: {e}. කරුණාකර Colab Files panel එකෙන් Manual Download කරගන්න.")

def _normalize_food_name(name: str) -> str:
    """
    ආහාර නාමයේ ඇති වරහන් (serving sizes) සහ අමතර සංකේත ඉවත් කර,
    Deduplication සඳහා පිරිසිදු නාමය ලබා දෙයි.
    උදා: 'woodapple juice (1 glass)' -> 'woodapple juice'
    """
    clean = re.sub(r'\(.*?\)', '', name).lower().strip()
    return clean

def generate_database_with_ai(model_id: str = AI_MODEL_NAME, target_items: int = 2500) -> Dict[str, Dict[str, Any]]:
    """
    AI Model එකක් හරහා ආහාර ලැයිස්තුව උත්පාදනය කරන ප්‍රධාන ශ්‍රිතය.
    """
    start_time = time.time()
    generated_db: Dict[str, Dict[str, Any]] = {}
    
    # කලින් Save කළ දත්ත ඇත්නම් ඒවා Load කරගැනීම (Resume capability)
    if os.path.exists(OUTPUT_FILE_PATH):
        try:
            with open(OUTPUT_FILE_PATH, "r", encoding="utf-8") as f:
                generated_db = json.load(f)
            print(f"🔄 කලින් උත්පාදනය කළ ආහාර {len(generated_db)} ක් නැවත Load කරගන්නා ලදී.")
        except Exception:
            pass

    # දැනට පවතින ආහාරවල Normalized නාම ලැයිස්තුව (Duplicate වැළැක්වීමට)
    existing_normalized_names = set(_normalize_food_name(k) for k in generated_db.keys())

    if not HAS_TRANSFORMERS or not torch.cuda.is_available():
        print("⚠️ [Warning] GPU හෝ Transformers library එක සොයාගත නොහැක.")
        print("💡 Google Colab හි Runtime -> Change runtime type -> T4 GPU තෝරා Run කරන්න.")
        print("🔄 දැනට සාම්පල දත්ත (Sample items) උත්පාදනය කරනු ලැබේ...\n")
        fallback_db = generate_sample_fallback_db()
        for k, v in fallback_db.items():
            norm_k = _normalize_food_name(k)
            if norm_k not in existing_normalized_names:
                generated_db[k] = v
                existing_normalized_names.add(norm_k)
        save_json_incrementally(generated_db, OUTPUT_FILE_PATH)
        return generated_db

    print(f"🚀 AI Model එක Load කරමින් පවතී: {model_id} (4-Bit Quantized for Colab Free Tier)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    items_per_cat = target_items // len(CATEGORIES)

    for idx, category in enumerate(CATEGORIES, 1):
        # කාල සීමාව (5 Hours) පරීක්ෂා කිරීම
        elapsed_secs = time.time() - start_time
        if elapsed_secs > MAX_EXECUTION_TIME_SECONDS:
            print(f"\n⏰ [Timeout Safety] පැය 5ක උපරිම කාල සීමාව ආසන්න බැවින් ක්‍රියාවලිය ආරක්ෂිතව නතර කරනු ලැබේ.")
            print(f"💾 මේ දක්වා සාර්ථකව උත්පාදනය කළ ආහාර {len(generated_db)} ක් ආරක්ෂිතව Save කර ඇත.")
            break

        print(f"[{idx}/{len(CATEGORIES)}] 🍛 Generating items for category: {category}...")
        
        # දැනට සාදා ඇති ආහාර වලින් සාම්පල 30ක් AI prompt එකට යැවීම (Duplicate වීම වැළැක්වීමට)
        sample_exclusions = ", ".join(list(existing_normalized_names)[-30:]) if existing_normalized_names else "None"
        exclusion_prompt = f"\nCRITICAL: Do NOT generate any items that are identical or very similar to these already created foods: {sample_exclusions}." if existing_normalized_names else ""

        prompt = f"""You are a certified Sri Lankan nutritionist. Generate exactly {items_per_cat} distinct Sri Lankan and South Asian food/beverage items belonging to the category: '{category}'.{exclusion_prompt}
Output STRICTLY a valid JSON object where keys are the food name with standard serving size in parentheses, and values contain calories (rounded to nearest 10), protein (g), carbs (g), and fat (g).
Example format:
{{
  "woodapple juice (1 glass)": {{"calories": 180, "protein": 2.0, "carbs": 42.0, "fat": 0.5}},
  "maalu paan / fish bun (1 medium)": {{"calories": 240, "protein": 8.0, "carbs": 32.0, "fat": 9.0}}
}}
Do NOT output any intro or outro markdown text, ONLY valid JSON."""

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1500, temperature=0.7)
            
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # JSON Extract & Parse with Deduplication
        try:
            json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
            batch_data = json.loads(json_str)
            new_count = 0
            dups_skipped = 0
            for k, v in batch_data.items():
                clean_key = k.lower().strip()
                norm_key = _normalize_food_name(clean_key)
                
                # Fuzzy Deduplication Check
                if norm_key in existing_normalized_names:
                    dups_skipped += 1
                    continue
                    
                if "calories" in v and "protein" in v:
                    generated_db[clean_key] = {
                        "calories": int(v.get("calories", 100)),
                        "protein": float(v.get("protein", 2.0)),
                        "carbs": float(v.get("carbs", 15.0)),
                        "fat": float(v.get("fat", 3.0))
                    }
                    existing_normalized_names.add(norm_key)
                    new_count += 1
            print(f"   ✨ අලුතින් ආහාර {new_count} ක් එක් කරන ලදී. (Skipped Duplicates: {dups_skipped} | මුළු සංඛ්‍යාව: {len(generated_db)})")
        except Exception as e:
            print(f"   ⚠️ Category {idx} parsing skipped due to formatting: {e}")
            
        # Incremental Save (සෑම කාණ්ඩයක් අවසානයේම Save වේ)
        save_json_incrementally(generated_db, OUTPUT_FILE_PATH)
            
    return generated_db

def generate_sample_fallback_db() -> Dict[str, Dict[str, Any]]:
    """
    GPU නොමැති විට ක්‍රියාත්මක වන සාම්පල දත්ත උත්පාදක ශ්‍රිතය.
    """
    sample_items = {
        "king coconut water (1 glass)": {"calories": 45, "protein": 0.5, "carbs": 11.0, "fat": 0.2},
        "woodapple juice (1 glass)": {"calories": 180, "protein": 2.0, "carbs": 42.0, "fat": 0.5},
        "passion fruit juice (1 glass)": {"calories": 120, "protein": 1.0, "carbs": 28.0, "fat": 0.2},
        "falooda (1 glass)": {"calories": 320, "protein": 6.0, "carbs": 52.0, "fat": 10.0},
        "plain tea with sugar (1 cup)": {"calories": 40, "protein": 0.0, "carbs": 10.0, "fat": 0.0},
        "milk tea / kiri thé (1 cup)": {"calories": 90, "protein": 3.0, "carbs": 12.0, "fat": 3.5},
        "gotukola kenda (1 bowl)": {"calories": 110, "protein": 3.0, "carbs": 20.0, "fat": 2.5},
        "karapincha kenda (1 bowl)": {"calories": 100, "protein": 2.5, "carbs": 18.0, "fat": 2.0},
        "fish bun / maalu paan (1 piece)": {"calories": 240, "protein": 8.0, "carbs": 32.0, "fat": 9.0},
        "seeni sambol bun (1 piece)": {"calories": 220, "protein": 5.0, "carbs": 38.0, "fat": 6.0},
        "chinese roll / fish roll (1 piece)": {"calories": 210, "protein": 6.0, "carbs": 24.0, "fat": 10.0},
        "vegetable roti (1 triangle)": {"calories": 190, "protein": 4.0, "carbs": 28.0, "fat": 7.0},
        "egg roti (1 piece)": {"calories": 250, "protein": 9.0, "carbs": 26.0, "fat": 12.0},
        "kokis (2 pieces)": {"calories": 150, "protein": 2.0, "carbs": 16.0, "fat": 9.0},
        "mung kavum (1 piece)": {"calories": 160, "protein": 3.0, "carbs": 28.0, "fat": 4.5},
        "kalu dodol (50g piece)": {"calories": 210, "protein": 2.0, "carbs": 35.0, "fat": 8.0},
        "thalaguli (2 pieces)": {"calories": 140, "protein": 3.0, "carbs": 15.0, "fat": 8.0},
        "masala dosa (1 medium)": {"calories": 350, "protein": 7.0, "carbs": 58.0, "fat": 10.0},
        "medu vada / ulundhu vadai (2 pieces)": {"calories": 260, "protein": 8.0, "carbs": 28.0, "fat": 13.0},
        "fried cassava chips (50g)": {"calories": 250, "protein": 1.5, "carbs": 32.0, "fat": 13.0},
        "devilled chicken (150g)": {"calories": 280, "protein": 24.0, "carbs": 12.0, "fat": 15.0},
        "black pork curry (150g)": {"calories": 340, "protein": 22.0, "carbs": 4.0, "fat": 26.0},
        "crab curry (200g)": {"calories": 210, "protein": 24.0, "carbs": 5.0, "fat": 10.0},
        "hot and sweet cuttlefish (150g)": {"calories": 230, "protein": 20.0, "carbs": 14.0, "fat": 10.0},
        "tempered kadala / chickpeas (1 cup)": {"calories": 260, "protein": 12.0, "carbs": 42.0, "fat": 5.0},
        "rambutan (5 fruits)": {"calories": 60, "protein": 1.0, "carbs": 14.0, "fat": 0.2},
        "mangosteen (3 fruits)": {"calories": 70, "protein": 1.0, "carbs": 17.0, "fat": 0.5},
        "red banana / rath kehel (1 medium)": {"calories": 110, "protein": 1.3, "carbs": 28.0, "fat": 0.3},
        "soursop / anoda juice (1 glass)": {"calories": 130, "protein": 1.5, "carbs": 30.0, "fat": 0.4}
    }
    return sample_items

if __name__ == "__main__":
    print("=====================================================================")
    print("🌟 AI Nutrition Database Generator Started 🌟")
    print("=====================================================================")
    
    start_t = time.time()
    db_results = generate_database_with_ai()
    
    elapsed = round(time.time() - start_t, 2)
    print(f"\n✅ ක්‍රියාවලිය අවසන්! ආහාර වර්ග {len(db_results)} ක් '{OUTPUT_FILE_PATH}' ගොනුවේ ආරක්ෂිතව Save කර ඇත.")
    print(f"⏱️ ගතවූ කාලය: {elapsed} තත්පර.")
    
    # Colab එක තුළ ක්‍රියාත්මක වේ නම් ස්වයංක්‍රීයව Laptop එකට Download කිරීම
    trigger_colab_auto_download(OUTPUT_FILE_PATH)
    
    print("💡 දැන් Backend එක (main.py) Run කළ විට මෙම ගොනුව ස්වයංක්‍රීයව කියවනු ලැබේ!")
