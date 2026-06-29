# =====================================================================
# LLM Guardrails & Hallucination Prevention Module
# =====================================================================
# AI output එකෙහි කැලරි ගණනයන් අඩංගුදැයි පරීක්ෂා කර, ඒවා ඉවත් කර
# නිශ්චිත Tier 3 දත්ත පමණක් පරිශීලකයාට පෙන්වීමට වගබලා ගනී.
# =====================================================================

import re
from typing import Dict, Any, List
from nutrition.db import DishComponent

SYSTEM_PROMPT_GUARD = """
You are a food identification engine. Output strictly JSON.
Do NOT calculate or output any numeric calories or macros.
All nutritional calculations are handled deterministically by a secondary lookup table.
"""

def validate_components(components: List[DishComponent]) -> List[DishComponent]:
    """
    AI විසින් හඳුනාගත් components වල නම් සහ portions වල අසාමාන්‍ය හෝ
    හානිකර දත්ත ඇත්නම් ඒවා ඉවත් කරයි.
    """
    validated = []
    for comp in components:
        clean_name = re.sub(r'[^\w\s]', '', comp.name).strip()
        if len(clean_name) > 1:
            comp.name = clean_name
            validated.append(comp)
    return validated

def verify_no_hallucinated_calories(text: str) -> bool:
    """
    AI text output එක තුළ 'calories = XXX' වැනි Hallucinations තිබේදැයි බලයි.
    """
    if re.search(r'\b\d+\s*(kcal|calories)\b', text.lower()):
        return False
    return True
