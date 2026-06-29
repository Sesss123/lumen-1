import os
import hashlib
import json
from typing import Tuple
from lumen.security import is_layer_enabled, security_config

# Try loading cryptography Ed25519 signing libraries. Fallback to SHA256 verification if missing.
try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    HAS_SIGNING = True
except ImportError:
    HAS_SIGNING = False

# Manifest directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
MANIFEST_PATH = os.path.join(DATA_DIR, "dataset_manifest.json")

def get_file_sha256(filepath: str) -> str:
    """Computes the SHA256 checksum hash of a file."""
    if not os.path.exists(filepath):
        return ""
        
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return ""

def update_dataset_manifest(filepath: str) -> None:
    """Updates the dataset manifest catalog with the SHA256 hash of the modified dataset file (Layer 11)."""
    if not os.path.exists(filepath):
        return
        
    filename = os.path.basename(filepath)
    file_hash = get_file_sha256(filepath)
    
    manifest_data = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                manifest_data = json.load(f)
        except Exception:
            pass
            
    manifest_data[filename] = file_hash
    
    try:
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest_data, f, indent=4)
    except Exception as e:
        print(f"⚠️ Failed to update dataset manifest: {e}")

def verify_dataset_integrity(filepath: str) -> Tuple[bool, str]:
    """
    Verifies that a dataset file matches the registered SHA256 hash in the manifest (Layer 11).
    """
    if not is_layer_enabled(11):
        return True, ""
        
    if not os.path.exists(filepath):
        return False, "Target dataset file not found."
        
    filename = os.path.basename(filepath)
    current_hash = get_file_sha256(filepath)
    
    if not os.path.exists(MANIFEST_PATH):
        # Create manifest automatically if missing (bootstrapping phase)
        update_dataset_manifest(filepath)
        return True, "Manifest was bootstrapped."
        
    try:
        with open(MANIFEST_PATH, "r") as f:
            manifest_data = json.load(f)
    except Exception as e:
        return False, f"Manifest read error: {e}"
        
    registered_hash = manifest_data.get(filename)
    if not registered_hash:
        # Register new file
        update_dataset_manifest(filepath)
        return True, "New file hash registered."
        
    if current_hash != registered_hash:
        return False, "Integrity Error: Dataset has been tampered with or corrupted (hash mismatch)."
        
    return True, ""

# --- MODEL CHECKPOINT INTEGRITY AND Ed25519 SIGNING ---

def generate_signing_keys() -> Tuple[str, str]:
    """Generates Ed25519 key pair for model weight checkpoint authentication."""
    if not HAS_SIGNING:
        return "", ""
        
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_bytes.decode('utf-8'), public_bytes.decode('utf-8')

def sign_checkpoint(checkpoint_dir: str) -> bool:
    """
    Generates SHA-256 hash list of weights in a checkpoint, signs it with Ed25519 private key (Layer 6/11).
    Saves signature to 'checkpoint.sig' and hashes to 'checkpoint.sha256'.
    """
    if not is_layer_enabled(11):
        return True
        
    sig_config = security_config.get("model_signing", {})
    priv_key_path = sig_config.get("private_key_path", "../checkpoints/signing_key.pem")
    
    # Resolve absolute path
    if not os.path.isabs(priv_key_path):
        priv_key_path = os.path.abspath(os.path.join(checkpoint_dir, priv_key_path))
        
    # Check if key pair needs to be generated
    if not os.path.exists(priv_key_path):
        if sig_config.get("generate_keys_if_missing", True) and HAS_SIGNING:
            try:
                os.makedirs(os.path.dirname(priv_key_path), exist_ok=True)
                priv_pem, pub_pem = generate_signing_keys()
                with open(priv_key_path, "w") as f:
                    f.write(priv_pem)
                pub_key_path = sig_config.get("public_key_path", "../checkpoints/signing_key.pub")
                if not os.path.isabs(pub_key_path):
                    pub_key_path = os.path.abspath(os.path.join(checkpoint_dir, pub_key_path))
                with open(pub_key_path, "w") as f:
                    f.write(pub_pem)
                print(f"🔑 Checkpoint Ed25519 keys generated at: {priv_key_path}")
            except Exception as e:
                print(f"⚠️ Failed to generate checkpoint keys: {e}")
        else:
            return False
            
    # Hash files
    weight_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.bin', '.safetensors', '.json', '.txt')) and f != 'checkpoint.sig' and f != 'checkpoint.sha256']
    manifest = {}
    for filename in weight_files:
        filepath = os.path.join(checkpoint_dir, filename)
        manifest[filename] = get_file_sha256(filepath)
        
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode('utf-8')
    
    # Save hashes
    with open(os.path.join(checkpoint_dir, "checkpoint.sha256"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    # Sign manifest
    if HAS_SIGNING:
        try:
            with open(priv_key_path, "rb") as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
            signature = private_key.sign(manifest_bytes)
            with open(os.path.join(checkpoint_dir, "checkpoint.sig"), "wb") as f:
                f.write(signature)
            return True
        except Exception as e:
            print(f"⚠️ Checkpoint signing failed: {e}")
            
    return False

def verify_checkpoint_integrity(checkpoint_dir: str) -> Tuple[bool, str]:
    """
    Verifies model checkpoints by validating file hashes and checking the Ed25519 signature (Layer 11).
    """
    if not is_layer_enabled(11):
        return True, "Validation bypassed: Layer 11 disabled."
        
    sha256_path = os.path.join(checkpoint_dir, "checkpoint.sha256")
    sig_path = os.path.join(checkpoint_dir, "checkpoint.sig")
    
    if not os.path.exists(sha256_path):
        return False, "Checkpoint SHA256 catalog is missing."
        
    try:
        with open(sha256_path, "r") as f:
            manifest = json.load(f)
            
        # 1. Verify weights hashes
        for filename, expected_hash in manifest.items():
            filepath = os.path.join(checkpoint_dir, filename)
            if not os.path.exists(filepath):
                return False, f"Checkpoint file '{filename}' is missing."
            actual_hash = get_file_sha256(filepath)
            if actual_hash != expected_hash:
                return False, f"Checkpoint weight '{filename}' has mismatching hash (tampered or corrupted)."
                
        # 2. Verify Ed25519 signature
        if HAS_SIGNING and os.path.exists(sig_path):
            sig_config = security_config.get("model_signing", {})
            pub_key_path = sig_config.get("public_key_path", "../checkpoints/signing_key.pub")
            if not os.path.isabs(pub_key_path):
                pub_key_path = os.path.abspath(os.path.join(checkpoint_dir, pub_key_path))
                
            if os.path.exists(pub_key_path):
                with open(pub_key_path, "rb") as f:
                    public_key = serialization.load_pem_public_key(f.read())
                with open(sig_path, "rb") as f:
                    signature = f.read()
                    
                manifest_bytes = json.dumps(manifest, sort_keys=True).encode('utf-8')
                try:
                    public_key.verify(signature, manifest_bytes)
                    return True, "Weights hash and Ed25519 signature verification successful."
                except Exception:
                    return False, "Weights verification failed: Invalid Ed25519 digital signature."
            else:
                return True, "Weights verified with SHA256 (Ed25519 public key missing)."
                
        return True, "Weights verified successfully using SHA256 hashes."
    except Exception as e:
        return False, f"Verification failed: {e}"
