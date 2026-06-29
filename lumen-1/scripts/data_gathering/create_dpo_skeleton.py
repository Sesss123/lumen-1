import json
import os

def main():
    dpo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dpo_data.jsonl"))
    os.makedirs(os.path.dirname(dpo_path), exist_ok=True)
    
    samples = [
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
    
    with open(dpo_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ Generated DPO skeleton dataset with {len(samples)} items in {dpo_path}!")

if __name__ == "__main__":
    main()
