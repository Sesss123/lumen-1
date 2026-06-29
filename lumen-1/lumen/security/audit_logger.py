import os
import hmac
import hashlib
import secrets
import time
from typing import List, Dict, Any, Tuple
from lumen.security import is_layer_enabled, security_config

# Resolve file path relative to this security package
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
LOG_FILE_PATH = os.path.join(DATA_DIR, "security_audit.log")

# Bug 6 Fix: Use a DEDICATED audit HMAC secret — never share with API key.
# If 'audit_hmac_secret' is not set in security.yaml, generate a cryptographically
# secure runtime secret. This means API key leaks cannot compromise audit log integrity.
_configured_secret = security_config.get("audit_hmac_secret", "")
if _configured_secret:
    AUDIT_SECRET = _configured_secret.encode('utf-8')
else:
    # Generate a fresh random secret per process — logs written in this session
    # will still chain correctly; cross-restart verification requires a persistent key.
    AUDIT_SECRET = secrets.token_hex(32).encode('utf-8')

ROOT_HMAC = "0" * 64


# Active security threat alert callbacks
on_log_callbacks = []

def get_last_signature() -> str:
    """Reads the last line of the log file to fetch the previous entry's HMAC signature."""
    if not os.path.exists(LOG_FILE_PATH) or os.path.getsize(LOG_FILE_PATH) == 0:
        return ROOT_HMAC
        
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return ROOT_HMAC
            # Get last line
            last_line = lines[-1].strip()
            # Format: timestamp | client_ip | endpoint | threat_level | message | hmac
            parts = last_line.split(" | ")
            if len(parts) >= 6:
                return parts[-1]
    except Exception:
        pass
    return ROOT_HMAC

def write_audit_log(client_ip: str, endpoint: str, threat_level: str, message: str) -> str:
    """
    Writes a new chained audit log entry using HMAC chain signature tamper protection (Layer 5/11).
    Returns the generated entry string.
    """
    if not is_layer_enabled(5) and not is_layer_enabled(11):
        return ""
        
    os.makedirs(DATA_DIR, exist_ok=True)
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    prev_sig = get_last_signature()
    
    # Payload to sign
    payload = f"{timestamp} {client_ip} {endpoint} {threat_level} {message} {prev_sig}"
    
    # Compute current HMAC signature
    signature = hmac.new(AUDIT_SECRET, payload.encode('utf-8'), hashlib.sha256).hexdigest()
    
    # Format entry line
    log_entry = f"{timestamp} | {client_ip} | {endpoint} | {threat_level} | {message} | {signature}\n"
    
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"⚠️ Failed to write security audit log: {e}")
        
    # Execute threat notifications (Improve 5)
    for cb in on_log_callbacks:
        try:
            cb(client_ip, endpoint, threat_level, message)
        except Exception:
            pass
            
    return log_entry.strip()

def verify_audit_log_integrity() -> Tuple[bool, List[Dict[str, Any]], int]:
    """
    Verifies the integrity of the audit log by re-hashing all entries in chain (Layer 11).
    Returns (is_intact, parsed_logs, tampered_line_number).
    """
    if not os.path.exists(LOG_FILE_PATH):
        return True, [], -1
        
    parsed_logs = []
    prev_sig = ROOT_HMAC
    
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(" | ")
                if len(parts) < 6:
                    # Malformed entry, signaling tampering or corruption
                    return False, parsed_logs, idx + 1
                    
                timestamp, client_ip, endpoint, threat_level, message, signature = parts[:6]
                
                # Verify HMAC hash signature chain
                payload = f"{timestamp} {client_ip} {endpoint} {threat_level} {message} {prev_sig}"
                expected_sig = hmac.new(AUDIT_SECRET, payload.encode('utf-8'), hashlib.sha256).hexdigest()
                
                if signature != expected_sig:
                    # Signature mismatch: log file tampered or edited
                    return False, parsed_logs, idx + 1
                    
                parsed_logs.append({
                    "line": idx + 1,
                    "timestamp": timestamp,
                    "client_ip": client_ip,
                    "endpoint": endpoint,
                    "threat_level": threat_level,
                    "message": message,
                    "signature": signature
                })
                prev_sig = signature
                
        return True, parsed_logs, -1
    except Exception as e:
        print(f"⚠️ Audit log integrity verification error: {e}")
        return False, [], 0
