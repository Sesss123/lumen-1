import os
import yaml

# Centralized config loader for lumen/security package
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "configs"))
CONFIG_PATH = os.path.join(CONFIGS_DIR, "security.yaml")

security_config = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            security_config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Error loading security.yaml: {e}")
else:
    # Default configs if file doesn't exist
    security_config = {
        "enabled_layers": list(range(1, 13)),
        "ip_allowlist_mode": False,
        "ip_allowlist": ["127.0.0.1", "::1"],
        "api_key": "lumen_default_secure_api_key_2026",
        "rate_limiting": {
            "limit": 10,
            "window": 60,
            "burst_limit": 20,
            "burst_window": 5,
            "burst_block_duration": 300
        },
        "request_size_limit": 10485760,
        "max_upload_size": 52428800,
        "session_idle_timeout": 1800,
        "hard_max_new_tokens": 512,
        "dlp": {
            "block_raw_downloads": True,
            "mask_checkpoints_dir": True
        }
    }

def is_layer_enabled(layer_num: int) -> bool:
    """Helper to check if a specific security layer is globally enabled."""
    enabled_layers = security_config.get("enabled_layers", list(range(1, 13)))
    return layer_num in enabled_layers
