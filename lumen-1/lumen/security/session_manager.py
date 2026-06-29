import secrets
import time
from typing import Dict, Set, Tuple
from lumen.security import is_layer_enabled, security_config

# Active sessions: {session_id: last_activity_timestamp}
active_sessions: Dict[str, float] = {}

# Replay attack prevention: {nonce: timestamp}
used_nonces: Dict[str, float] = {}

def create_session() -> str:
    """Creates a cryptographically secure session token (Layer 7)."""
    token = secrets.token_urlsafe(32)
    session_id = f"sess_{token}"
    active_sessions[session_id] = time.time()
    return session_id

def validate_session(session_id: str) -> bool:
    """Checks if a session is valid and updates its activity timestamp (Layer 7)."""
    if not is_layer_enabled(7):
        return True
        
    if not session_id or session_id not in active_sessions:
        return False
        
    now = time.time()
    last_active = active_sessions[session_id]
    timeout = security_config.get("session_idle_timeout", 1800) # Default 30 min
    
    if now - last_active > timeout:
        # Session expired
        del active_sessions[session_id]
        return False
        
    # Update activity timestamp
    active_sessions[session_id] = now
    return True

def validate_nonce(nonce: str, request_timestamp: float) -> Tuple[bool, str]:
    """
    Validates requests nonces to prevent Replay Attacks (Layer 7).
    Nonces are only valid within a 5-minute time window.
    """
    if not is_layer_enabled(7):
        return True, ""
        
    if not nonce:
        return False, "Security Error: Missing request nonce."
        
    now = time.time()
    
    # 1. Prevent requests with stale timestamps (>5 minutes clock skew)
    if abs(now - request_timestamp) > 300:
        return False, "Security Error: Stale request timestamp skew detected."
        
    # 2. Check if nonce has already been used
    if nonce in used_nonces:
        return False, "Security Error: Replay attack detected. Nonce already consumed."
        
    # 3. Register nonce
    used_nonces[nonce] = now
    
    # Clean up old nonces to prevent memory leak
    stale_nonces = [n for n, ts in used_nonces.items() if now - ts > 300]
    for n in stale_nonces:
        if n in used_nonces:
            del used_nonces[n]
        
    return True, ""

# --- PERIODIC BACKGROUND CLEANUP SYSTEM (Improve 2) ---
import threading

def cleanup_expired_sessions():
    """Purges expired sessions, nonces, and stale intent histories (Layer 7 & 8)."""
    now = time.time()
    
    # 1. Purge active_sessions
    timeout = security_config.get("session_idle_timeout", 1800)
    expired_sess = [s for s, ts in active_sessions.items() if now - ts > timeout]
    for s in expired_sess:
        if s in active_sessions:
            del active_sessions[s]
            
    # 2. Purge stale nonces
    expired_nonces = [n for n, ts in used_nonces.items() if now - ts > 300]
    for n in expired_nonces:
        if n in used_nonces:
            del used_nonces[n]
            
    # 3. Purge intent classifier session histories (> 2 hours)
    try:
        from lumen.security import intent_classifier
        intent_classifier.purge_stale_sessions(7200)
    except Exception as e:
        print(f"⚠️ Failed to purge intent classifier sessions in background: {e}")

    # 4. Bug 8 Fix: Purge stale IP entries from rate limiter request_history
    #    Entries with no activity in the past 60s are removed to prevent RAM growth.
    try:
        from lumen.security.rate_limiter import cleanup_stale_ips
        removed = cleanup_stale_ips(max_inactivity_seconds=60.0)
        if removed > 0:
            print(f"🧹 Rate limiter cleanup: removed {removed} stale IP entries.")
    except Exception as e:
        print(f"⚠️ Failed to cleanup stale rate limiter IPs: {e}")

def start_cleanup_daemon():
    def cleanup_loop():
        while True:
            try:
                time.sleep(900)  # Every 15 minutes
                cleanup_expired_sessions()
            except Exception:
                pass
                
    t = threading.Thread(target=cleanup_loop, name="LumenSecurityCleanup", daemon=True)
    t.start()

# Start background daemon on module load
start_cleanup_daemon()

