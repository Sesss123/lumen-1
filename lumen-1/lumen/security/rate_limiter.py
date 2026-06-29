import time
from typing import Dict, List, Tuple
from lumen.security import is_layer_enabled, security_config

# IP history log
# {ip: [timestamp1, timestamp2, ...]}
request_history: Dict[str, List[float]] = {}

# Blocked IPs
# {ip: block_expiration_timestamp}
blocked_ips: Dict[str, float] = {}

def get_request_fingerprint(client_ip: str, headers: dict) -> str:
    """Generates a request fingerprint using IP and User-Agent to prevent IP-spoof/VPN limit bypasses (Layer 9)."""
    user_agent = headers.get("user-agent", "unknown")
    # Return a unique string signature
    import hashlib
    fingerprint = hashlib.sha256(f"{client_ip}:{user_agent}".encode('utf-8')).hexdigest()[:16]
    return fingerprint

def check_rate_limit(client_ip: str) -> Tuple[bool, str]:
    """
    Validates per-IP sliding window rate limits, burst abuse, and block statuses (Layer 2).
    Returns (is_allowed, error_message).
    """
    if not is_layer_enabled(2):
        return True, ""
        
    now = time.time()
    
    # 1. Check if IP is currently blocked
    if client_ip in blocked_ips:
        block_time = blocked_ips[client_ip]
        if now < block_time:
            remaining = int(block_time - now)
            return False, f"Abuse Detected: IP temporarily blocked. Try again in {remaining} seconds."
        else:
            # Block expired
            del blocked_ips[client_ip]
            
    # 2. Record request and purge old entries
    if client_ip not in request_history:
        request_history[client_ip] = []
        
    # Get config settings
    rl_config = security_config.get("rate_limiting", {})
    window = rl_config.get("window", 60)
    limit = rl_config.get("limit", 10)
    burst_window = rl_config.get("burst_window", 5)
    burst_limit = rl_config.get("burst_limit", 20)
    block_duration = rl_config.get("burst_block_duration", 300)
    
    # Keep only timestamps inside 60 second window
    request_history[client_ip] = [t for t in request_history[client_ip] if now - t < window]
    request_history[client_ip].append(now)
    
    # 3. Burst abuse check (e.g. >20 requests in 5 seconds)
    burst_history = [t for t in request_history[client_ip] if now - t < burst_window]
    if len(burst_history) >= burst_limit:
        blocked_ips[client_ip] = now + block_duration
        return False, f"Abuse Detected: Request burst threshold exceeded. IP blocked for {block_duration // 60} minutes."
        
    # 4. Standard rate check (>10 requests in 60 seconds)
    if len(request_history[client_ip]) > limit:
        return False, f"Too many requests. Limit is {limit} requests per minute."
        
    return True, ""

def enforce_gpu_abuse_guard(max_new_tokens: int) -> int:
    """Enforces a hard limit on maximum generated tokens to prevent resource exhaustion (Layer 2)."""
    if not is_layer_enabled(2):
        return max_new_tokens

    hard_limit = security_config.get("hard_max_new_tokens", 512)
    return min(max_new_tokens, hard_limit)


def cleanup_stale_ips(max_inactivity_seconds: float = 60.0) -> int:
    """
    Bug 8 Fix: Purges IP entries from request_history that have had no
    activity for more than `max_inactivity_seconds` (default 60s).
    This prevents unbounded RAM growth on long-running servers.
    Call this from the session cleanup daemon or a periodic background task.
    Returns the number of IPs removed.
    """
    now = time.time()
    stale_ips = [
        ip for ip, timestamps in request_history.items()
        if not timestamps or (now - max(timestamps)) > max_inactivity_seconds
    ]
    for ip in stale_ips:
        del request_history[ip]
        # Also remove from blocked_ips if block has already expired
        if ip in blocked_ips and now >= blocked_ips[ip]:
            del blocked_ips[ip]
    return len(stale_ips)

