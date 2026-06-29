import os
import re
from typing import Tuple
from lumen.security import is_layer_enabled, security_config

# Default allowed extensions whitelist (documents / data)
ALLOWED_EXTENSIONS = {'.csv', '.json', '.jsonl', '.txt', '.md', '.pdf'}

# Bug 9 Fix: Multimodal-specific allowed extensions (video & audio)
MULTIMODAL_EXTENSIONS = {
    '.mp4', '.webm', '.mov', '.avi',       # video
    '.wav', '.mp3', '.ogg', '.m4a', '.flac',  # audio
}

def is_safe_path(base_dir: str, path: str, follow_symlinks: bool = True) -> bool:
    """Verifies that the target path resolves inside the base directory to prevent Path Traversal (Layer 3)."""
    if not is_layer_enabled(3):
        return True

    if follow_symlinks:
        matchpath = os.path.realpath(path)
        base = os.path.realpath(base_dir)
    else:
        matchpath = os.path.abspath(path)
        base = os.path.abspath(base_dir)
    return matchpath.startswith(base + os.sep) or matchpath == base

def sanitize_upload_filename(filename: str) -> str:
    """Sanitizes filename to remove path traversal characters and non-alphanumeric symbols (Layer 3)."""
    # Get basename to prevent path tricks
    base = os.path.basename(filename)
    # Remove everything except alphanumeric, dot, dashes, and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\._\-]', '', base)
    return sanitized

def validate_file_upload(filename: str, file_size: int, context: str = "default") -> Tuple[bool, str]:
    """
    Validates file upload extension, name, and size constraints (Layer 3).
    Bug 9 Fix: Added `context` parameter.
      - context="multimodal" → allows video/audio extensions for the multimodal endpoint.
      - context="default"    → uses standard document/data extension whitelist.
    """
    if not is_layer_enabled(3):
        return True, ""

    # 1. Check size limit
    max_size = security_config.get("max_upload_size", 52428800)  # Default 50MB
    if file_size > max_size:
        return False, f"File size exceeds the maximum limit of {max_size // (1024*1024)}MB."

    # 2. Select the appropriate extension whitelist based on context
    if context == "multimodal":
        allowed = ALLOWED_EXTENSIONS | MULTIMODAL_EXTENSIONS
    else:
        allowed = ALLOWED_EXTENSIONS

    ext = os.path.splitext(filename.lower())[1]
    if ext not in allowed:
        return False, f"File extension '{ext}' is not supported. Whitelisted: {sorted(allowed)}"

    # 3. Check for path traversal signs
    if '..' in filename or '/' in filename or '\\' in filename:
        return False, "Path traversal characters detected in upload filename."

    return True, ""


def validate_mime_content(file_path: str) -> Tuple[bool, str]:
    """
    Checks the file magic bytes and internal structures to verify it doesn't contain
    binary executables, null-byte payloads, or script tags (Layer 3).
    """
    if not is_layer_enabled(3):
        return True, ""
        
    if not os.path.exists(file_path):
        return False, "Target validation file does not exist."
        
    try:
        # Read the first 4096 bytes to inspect magic headers
        with open(file_path, "rb") as f:
            header_bytes = f.read(4096)
            
        # 1. Null byte detection (indicates binary payload in text format)
        if b'\x00' in header_bytes:
            return False, "Malicious Content Blocked: Binary payload/null-bytes detected in file."
            
        # 2. Inspect magic headers for common executables
        # PE EXE: MZ header
        if header_bytes.startswith(b'MZ'):
            return False, "Malicious Content Blocked: Executable PE binary format detected."
        # ELF binary: \x7fELF
        if header_bytes.startswith(b'\x7fELF'):
            return False, "Malicious Content Blocked: Executable ELF format detected."
        # ZIP/Jar/Office format mismatch if expecting plain text csv
        if header_bytes.startswith(b'PK\x03\x04') and not file_path.lower().endswith(('.zip', '.docx', '.xlsx')):
            ext = os.path.splitext(file_path.lower())[1]
            if ext in ['.csv', '.txt', '.md', '.json', '.jsonl']:
                return False, f"MIME Conflict: Compressed binary structure (PK) found in '{ext}' text file."
                
        # 3. Embedded Script/HTML Injection Check (for text formats like CSV/JSON)
        # Verify that CSV/JSON uploads don't contain embedded HTML scripts/forms
        text_preview = header_bytes.decode('utf-8', errors='ignore')
        suspicious_tags = [r'<script\b', r'javascript:', r'<iframe\b', r'onload\s*=']
        for tag in suspicious_tags:
            if re.search(tag, text_preview, re.IGNORECASE):
                return False, f"Malicious Content Blocked: HTML/Javascript script tag injections found."
                
        return True, ""
    except Exception as e:
        return False, f"Validation failure during content integrity scan: {str(e)}"
