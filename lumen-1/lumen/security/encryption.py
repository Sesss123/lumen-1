import hmac
import hashlib
import base64
import os
from typing import Tuple, Optional
from lumen.security import is_layer_enabled, security_config

# Layer 7: End-to-End Cryptography and HMAC Request Signing

# Try loading cryptography library. Graceful degradation if missing.
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

SECRET_HMAC_KEY = security_config.get("api_key", "lumen_secure_key_2026").encode('utf-8')

# Active sessions keys mapping: {session_id: derived_shared_key_bytes}
active_session_keys = {}

def is_crypto_available() -> bool:
    return HAS_CRYPTO

def generate_ecdh_keypair() -> Tuple[str, object]:
    """Generates an ephemeral ECDH private key and returns its public key in hex format."""
    if not HAS_CRYPTO:
        return "", None
        
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    
    # Serialize public key to DER/RAW hex format
    pub_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    )
    return pub_bytes.hex(), private_key

def derive_shared_key(private_key_obj: object, peer_pub_hex: str) -> bytes:
    """Derives a shared symmetric key using private key and client's public key hex."""
    if not HAS_CRYPTO or private_key_obj is None:
        return b""
        
    try:
        peer_bytes = bytes.fromhex(peer_pub_hex)
        peer_public = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), peer_bytes)
        shared_key = private_key_obj.exchange(ec.ECDH(), peer_public)
        
        # Derive key using HKDF
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"lumen-session-encryption",
        ).derive(shared_key)
        
        return derived_key
    except Exception as e:
        print(f"⚠️ ECDH shared key derivation failed: {e}")
        return b""

def encrypt_aes_gcm(key: bytes, plaintext: str) -> str:
    """Encrypts plaintext using AES-256-GCM. Returns base64 encoded ciphertext (nonce + payload)."""
    if not HAS_CRYPTO or not key:
        return plaintext
        
    try:
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
        # Pack nonce and ciphertext together
        packed = nonce + ciphertext
        return base64.b64encode(packed).decode('utf-8')
    except Exception as e:
        print(f"⚠️ AES GCM encryption failed: {e}")
        return plaintext

def decrypt_aes_gcm(key: bytes, ciphertext_b64: str) -> str:
    """Decrypts base64 encoded AES-256-GCM ciphertext."""
    if not HAS_CRYPTO or not key:
        return ciphertext_b64
        
    try:
        packed = base64.b64decode(ciphertext_b64)
        if len(packed) < 12:
            return ciphertext_b64
        nonce = packed[:12]
        ciphertext = packed[12:]
        aesgcm = AESGCM(key)
        decrypted = aesgcm.decrypt(nonce, ciphertext, None)
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"⚠️ AES GCM decryption failed: {e}")
        raise ValueError("Decryption failed. Invalid payload or corrupted cipher key.")

def generate_request_signature(payload_bytes: bytes) -> str:
    """Generates HMAC-SHA256 signature for request payloads."""
    return hmac.new(SECRET_HMAC_KEY, payload_bytes, hashlib.sha256).hexdigest()

def verify_request_signature(payload_bytes: bytes, signature: str) -> bool:
    """
    Verifies HMAC request payload signature using constant-time comparison (Layer 7/12).
    """
    if not is_layer_enabled(7):
        return True
    if not signature:
        return False
        
    expected_sig = generate_request_signature(payload_bytes)
    # constant-time comparison to prevent timing attacks (Layer 12)
    return hmac.compare_digest(expected_sig, signature)
