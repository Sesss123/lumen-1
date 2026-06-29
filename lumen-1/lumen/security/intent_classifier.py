import re
import time
from typing import List, Dict, Tuple, Optional
from lumen.security import is_layer_enabled
# Bug 5 Fix: Import centralized persona hijack patterns so both modules are in sync
from lumen.security.guards import PERSONA_HIJACK_PATTERNS

# In-memory conversation history dictionary: {session_id: [query_history_list]}
session_history: Dict[str, List[str]] = {}
# Session activity tracking to prevent memory leak (Layer 8)
session_activity: Dict[str, float] = {}

# Rule-based threat scoring dictionary
SEMANTIC_THREAT_KEYWORDS = {
    # Jailbreak / DAN intent
    r"\b(?:dan|jailbreak|devmode|bypass|override)\b": 40,
    # Instruction ignoring
    r"\bignore\b.*\binstructions\b": 35,
    r"\bsystem\s+prompt\b": 30,
    # Cyberattack / credential theft
    r"\b(?:hack|ddos|malware|steal|bypass\s+auth|sql\s+injection|xss|phish)\b": 45,
    # Social engineering / authority hacking
    r"\byou\s+are\s+now\s+a\b": 25,
    r"\bact\s+as\b": 20,
    # Persona hijack patterns — Bug 5: shared from guards.py
    r"pretend\s+(?:you\s+are|to\s+be)": 30,
    r"disregard\s+(?:previous|all|the)\s+instructions?": 35,
    r"from\s+now\s+on\s+(?:you\s+are|you're|call\s+yourself)": 30,
    r"enable\s+developer\s+mode": 40,
    r"simulation\s+mode": 25,
}

# Bug 11 Fix: Harmful/toxic content keywords — scored separately for HARMFUL_CONTENT classification
HARMFUL_CONTENT_KEYWORDS = [
    r"\b(?:explosive|bomb|ied|pipe\s+bomb|grenade|bioweapon|nerve\s+agent)\b",
    r"\b(?:how\s+to\s+(?:make|build|create|synthesize)\s+(?:bomb|weapon|poison|drug))\b",
    r"\b(?:child\s+(?:abuse|porn|exploit)|csam|pedophil)\b",
    r"\b(?:self.?harm|suicide\s+method|kill\s+myself|how\s+to\s+die)\b",
    r"\b(?:racial\s+slur|hate\s+speech|ethnic\s+cleansing|genocide\s+(?:support|promote))\b",
    r"\b(?:sexual\s+assault|rape\s+(?:guide|how|method)|drug\s+rape)\b",
    r"\b(?:human\s+trafficking|sell\s+(?:people|humans)|sex\s+slave)\b",
    r"\b(?:doxx(?:ing)?|deanonymize|expose\s+personal\s+(?:info|address|location))\b",
]

def analyze_intent_heuristics(prompt: str) -> Tuple[str, int]:
    """
    Calculates threat score and classifies semantic intent based on heuristics (Layer 8).
    Bug 11 Fix: Checks HARMFUL_CONTENT_KEYWORDS first — toxic/weapons content now
    gets its own classification tier instead of falling through to SAFE.
    """
    score = 0
    normalized = prompt.lower()

    # Bug 11: Check for harmful/toxic content BEFORE generic scoring
    for hpattern in HARMFUL_CONTENT_KEYWORDS:
        if re.search(hpattern, normalized, re.IGNORECASE):
            return "HARMFUL_CONTENT", 100  # Always highest severity

    # Bug 5: Check persona hijack patterns using the centralized list from guards.py
    for hj_pattern in PERSONA_HIJACK_PATTERNS:
        if re.search(hj_pattern, normalized, re.IGNORECASE | re.DOTALL):
            score += 30
            break  # One match is enough to count persona hijack

    for pattern, weight in SEMANTIC_THREAT_KEYWORDS.items():
        if re.search(pattern, normalized):
            score += weight

    if score >= 60:
        return "INJECTION_ATTEMPT", score
    elif score >= 40:
        return "JAILBREAK", score
    elif score >= 20:
        return "SUSPICIOUS", score
    return "SAFE", score

def purge_stale_sessions(max_inactivity_seconds: float = 7200) -> int:
    """Purges sessions that have been inactive for more than `max_inactivity_seconds` (default 2 hours) to prevent memory leak (Layer 8)."""
    now = time.time()
    stale_sessions = [s for s, ts in session_activity.items() if now - ts > max_inactivity_seconds]
    for s in stale_sessions:
        if s in session_history:
            del session_history[s]
        if s in session_activity:
            del session_activity[s]
    return len(stale_sessions)

def update_and_analyze_multi_turn(session_id: str, prompt: str) -> Tuple[str, List[str]]:
    """
    Maintains a sliding 5-turn conversation history and detects multi-turn escalation attacks (Layer 8).
    Tracks session timestamps to prune expired session states and prevent memory leaks.
    Returns (status, history).
    """
    if not is_layer_enabled(8):
        return "SAFE", []
        
    if not session_id:
        return "SAFE", []
        
    now = time.time()
    # 1. Update activity timestamp
    session_activity[session_id] = now
    
    # 2. Prune old session states (>30 minutes / 1800 seconds of inactivity)
    stale_sessions = [s for s, ts in session_activity.items() if now - ts > 1800]
    for s in stale_sessions:
        if s in session_history:
            del session_history[s]
        if s in session_activity:
            del session_activity[s]
        
    if session_id not in session_history:
        session_history[session_id] = []
        
    # Append latest query
    session_history[session_id].append(prompt)
    
    # Keep sliding window of last 5 turns
    if len(session_history[session_id]) > 5:
        session_history[session_id].pop(0)
        
    # Analyze multi-turn patterns (e.g. cumulative suspect score)
    cumulative_score = 0
    suspicious_turns = 0
    
    for turn in session_history[session_id]:
        _, score = analyze_intent_heuristics(turn)
        if score > 0:
            cumulative_score += score
            suspicious_turns += 1
            
    # If the user has made 3+ suspicious queries in a row, or cumulative score is high, flag it
    if suspicious_turns >= 3 or cumulative_score >= 50:
        return "INJECTION_ATTEMPT", session_history[session_id]
    elif suspicious_turns >= 2 or cumulative_score >= 30:
        return "SUSPICIOUS", session_history[session_id]
    # Bug 11: Escalate HARMFUL_CONTENT from single turn to multi-turn status
    for turn in session_history[session_id]:
        cls, _ = analyze_intent_heuristics(turn)
        if cls == "HARMFUL_CONTENT":
            return "HARMFUL_CONTENT", session_history[session_id]
        
    return "SAFE", session_history[session_id]

def classify_prompt_intent(prompt: str, session_id: Optional[str] = None) -> dict:
    """
    Integrates Layer 8 intent checks and returns a safety classification report.
    """
    if not is_layer_enabled(8):
        return {"status": "SAFE", "score": 0, "multi_turn_status": "SAFE"}
        
    status, score = analyze_intent_heuristics(prompt)
    
    multi_turn_status = "SAFE"
    if session_id:
        multi_turn_status, _ = update_and_analyze_multi_turn(session_id, prompt)
        
    # Pick the highest threat status between direct query and multi-turn escalation
    # Priority: HARMFUL_CONTENT > INJECTION_ATTEMPT > JAILBREAK > SUSPICIOUS > SAFE
    severity_rank = {"SAFE": 0, "SUSPICIOUS": 1, "JAILBREAK": 2, "INJECTION_ATTEMPT": 3, "HARMFUL_CONTENT": 4}
    final_status = status
    if severity_rank.get(multi_turn_status, 0) > severity_rank.get(status, 0):
        final_status = multi_turn_status

    return {
        "status": final_status,
        "score": score,
        "multi_turn_status": multi_turn_status
    }
