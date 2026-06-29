import os
from typing import List, Dict, Any
from lumen.security import is_layer_enabled, security_config

def filter_database_response(raw_records: list) -> list:
    """
    Implements DLP Layer 10: Isolates raw database records and returns a filtered/safe
    view of the data. Removes potential internal system columns or sensitive metadata.
    Bug 10 Fix: lat/lng GPS fields are approximated to 2 decimal places (~1.1km precision)
    instead of being exposed with full precision.
    """
    if not is_layer_enabled(10):
        return raw_records

    filtered = []
    # Strip any potential hidden/system tags or private properties
    sensitive_keys = {"_raw", "owner_id", "internal_notes", "created_by"}

    # GPS fields that must be truncated to protect exact location privacy
    gps_precision_keys = {"lat", "lng", "latitude", "longitude"}

    for record in raw_records:
        clean_record = {}
        for k, v in record.items():
            if k in sensitive_keys:
                continue  # Strip completely
            if k in gps_precision_keys and isinstance(v, (int, float)):
                # Bug 10 Fix: Approximate to 2 decimal places (~1.1km resolution)
                clean_record[k] = round(float(v), 2)
            else:
                clean_record[k] = v
        filtered.append(clean_record)

    return filtered


def is_checkpoint_download_allowed(filename: str) -> bool:
    """
    DLP Layer 10 Check: Blocks direct downloads of model checkpoint files (adapter weights, bins)
    to protect weights IP. Only allows downloading simple logs/metadata.
    """
    if not is_layer_enabled(10):
        return True
        
    # Get extension
    ext = os.path.splitext(filename.lower())[1]
    # Block model weights formats
    blocked_extensions = {".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".gguf"}
    if ext in blocked_extensions:
        return False
        
    return True
