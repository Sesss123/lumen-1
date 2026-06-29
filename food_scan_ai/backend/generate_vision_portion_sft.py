# =====================================================================
# Synthetic Vision & Portion Size SFT Dataset Generator
# =====================================================================
# මෙම ස්ක්‍රිප්ට් එක මගින් ආහාර පින්තූර විශ්ලේෂණය කර, පිඟානේ ප්‍රමාණය හා ඝනකම
# අනුව කෑම ප්‍රමාණයන් (Portion Sizes) ගණනය කරන 'Thinking Knowledge' (<think>)
# සහිත පුහුණු දත්ත (SFT Dataset) JSONL ආකාරයට උත්පාදනය කරයි.
# =====================================================================

import json
import random
import os
import time

DATABASE_PATH = "sri_lankan_nutrition_2500.json"
OUTPUT_JSONL_PATH = "vision_portion_sft_data.jsonl"
NUM_SAMPLES_TO_GENERATE = 500

# Sample fallback database if json not generated yet
FALLBACK_FOODS = {
    "white rice": {"calories": 200, "protein": 4.0, "carbs": 44.0, "fat": 0.5},
    "red rice": {"calories": 180, "protein": 4.5, "carbs": 40.0, "fat": 1.0},
    "chicken curry": {"calories": 180, "protein": 18.0, "carbs": 4.0, "fat": 10.0},
    "dhal curry / parippu": {"calories": 110, "protein": 6.0, "carbs": 15.0, "fat": 3.0},
    "pol sambol": {"calories": 75, "protein": 1.0, "carbs": 3.0, "fat": 6.0},
    "gotukola mallum": {"calories": 45, "protein": 2.5, "carbs": 6.0, "fat": 1.5},
    "fish ambul thiyal": {"calories": 140, "protein": 22.0, "carbs": 2.0, "fat": 5.0},
    "chicken kottu": {"calories": 650, "protein": 24.0, "carbs": 75.0, "fat": 28.0},
    "egg hopper": {"calories": 160, "protein": 7.0, "carbs": 18.0, "fat": 6.5},
    "plain hopper": {"calories": 90, "protein": 2.0, "carbs": 16.0, "fat": 2.0},
    "king coconut water": {"calories": 45, "protein": 0.5, "carbs": 11.0, "fat": 0.2},
    "woodapple juice": {"calories": 180, "protein": 2.0, "carbs": 42.0, "fat": 0.5},
    "masala dosa": {"calories": 350, "protein": 7.0, "carbs": 58.0, "fat": 10.0},
    "fried cassava chips": {"calories": 250, "protein": 1.5, "carbs": 32.0, "fat": 13.0}
}

PLATE_TYPES = [
    {"name": "Standard 25cm Ceramic Dining Plate", "area_cm2": 490},
    {"name": "Large 28cm Banquet Plate", "area_cm2": 615},
    {"name": "Traditional Banana Leaf (30cm serving area)", "area_cm2": 700},
    {"name": "Medium 20cm Breakfast Bowl", "area_cm2": 314},
    {"name": "Tall 350ml Glass", "volume_ml": 350}
]

def load_food_db():
    if os.path.exists(DATABASE_PATH):
        try:
            with open(DATABASE_PATH, "r", encoding="utf-8") as f:
                db = json.load(f)
                if len(db) > 10:
                    print(f"✅ Loaded {len(db)} food items from {DATABASE_PATH}")
                    return db
        except Exception as e:
            print(f"⚠️ Error loading database: {e}")
    print("💡 Using built-in fallback food items.")
    return FALLBACK_FOODS

def generate_thinking_sample(food_db):
    plate = random.choice(PLATE_TYPES)
    all_foods = list(food_db.keys())
    
    # Decide if it's a beverage or plate meal
    is_beverage = "Glass" in plate["name"]
    
    if is_beverage:
        bev_foods = [f for f in all_foods if "juice" in f.lower() or "water" in f.lower() or "tea" in f.lower() or "kenda" in f.lower()]
        selected_item = random.choice(bev_foods) if bev_foods else random.choice(all_foods)
        portion_str = "1 glass (300ml)"
        mult = 1.0
        
        user_prompt = f"Identify the item in this image scan and estimate its portion size and nutrition. Camera metadata: {plate['name']} filled to ~85% capacity."
        
        think_text = (
            f"1. Visual Reference Analysis: The image shows a {plate['name']}.\n"
            f"2. Volume Estimation: The liquid level reaches roughly 85% of the total glass capacity (~300ml).\n"
            f"3. Identification: The texture and color identify this as '{selected_item}'.\n"
            f"4. Nutritional Calculation: Standard serving (1 glass / 300ml) provides approximately "
            f"{food_db[selected_item].get('calories', 100)} kcal."
        )
        
        components = [{
            "name": selected_item,
            "estimated_portion": portion_str,
            "calories": round(food_db[selected_item].get('calories', 100) * mult, 1),
            "protein": round(food_db[selected_item].get('protein', 2.0) * mult, 1),
            "carbs": round(food_db[selected_item].get('carbs', 15.0) * mult, 1),
            "fat": round(food_db[selected_item].get('fat', 0.5) * mult, 1)
        }]
        dish_name = selected_item.title()
        
    else:
        # Plate meal with 2 to 4 components
        num_items = random.randint(2, 4)
        selected_items = random.sample(all_foods, min(num_items, len(all_foods)))
        
        user_prompt = f"Analyze this food image scan. Identify all visible items on the {plate['name']} and calculate exact portion sizes (grams/cups) and nutritional values."
        
        think_lines = [f"1. Reference Frame: Detected a {plate['name']} (approx surface area {plate.get('area_cm2', 500)} cm²)."]
        components = []
        
        # Divide plate percentages
        pcts = [random.randint(20, 50) for _ in range(len(selected_items))]
        total_pct = sum(pcts)
        norm_pcts = [int((p / total_pct) * 90) for p in pcts] # leave 10% empty space
        
        for idx, item in enumerate(selected_items):
            pct = norm_pcts[idx]
            est_grams = int(pct * 4.5) # rough visual mapping
            if est_grams < 30: est_grams = 30
            
            portion_str = f"{est_grams}g"
            if "rice" in item.lower():
                portion_str = f"{round(est_grams/180, 1)} cup ({est_grams}g)"
            elif "hopper" in item.lower() or "roti" in item.lower() or "dosa" in item.lower():
                portion_str = f"{max(1, round(est_grams/70))} pieces ({est_grams}g)"
                
            mult = est_grams / 150.0 # base normalized to 150g serving
            
            cal = round(food_db[item].get('calories', 150) * mult, 1)
            prot = round(food_db[item].get('protein', 5.0) * mult, 1)
            carb = round(food_db[item].get('carbs', 20.0) * mult, 1)
            fat = round(food_db[item].get('fat', 5.0) * mult, 1)
            
            think_lines.append(
                f"{idx+2}. Item '{item}': Occupies ~{pct}% of plate surface area with ~2.5cm depth. "
                f"Estimated mass: {est_grams}g. Calculated calories: ~{cal} kcal."
            )
            
            components.append({
                "name": item,
                "estimated_portion": portion_str,
                "calories": cal,
                "protein": prot,
                "carbs": carb,
                "fat": fat
            })
            
        think_text = "\n".join(think_lines)
        dish_name = f"Sri Lankan {selected_items[0].title()} Combo Plate"
        
    total_cals = round(sum(c["calories"] for c in components), 1)
    
    response_json = {
        "dish_name": dish_name,
        "confidence": round(random.uniform(0.91, 0.98), 2),
        "total_calories": total_cals,
        "components": components
    }
    
    assistant_response = f"<think>\n{think_text}\n</think>\n```json\n{json.dumps(response_json, indent=2)}\n```"
    
    # Format as ShareGPT conversation for SFT training
    sample = {
        "messages": [
            {"role": "system", "content": "You are Antigravity Food Vision AI. Analyze food scans, reason about portion sizes using visual geometry, and output precise nutritional JSON."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }
    return sample

def main():
    print("=====================================================================")
    print("🧠 Starting Synthetic Vision & Portion Size SFT Dataset Generator")
    print("=====================================================================")
    
    start_t = time.time()
    food_db = load_food_db()
    
    samples = []
    for _ in range(NUM_SAMPLES_TO_GENERATE):
        samples.append(generate_thinking_sample(food_db))
        
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    elapsed = round(time.time() - start_t, 2)
    print(f"\n✅ Successfully generated {len(samples)} training samples!")
    print(f"📁 Saved SFT dataset to: {OUTPUT_JSONL_PATH}")
    print(f"⏱️ Time taken: {elapsed} seconds.")
    print("\n💡 You can now use this dataset in Colab to train Qwen / Vision models!")

if __name__ == "__main__":
    main()
