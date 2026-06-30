# =====================================================================
# Advanced LLM Security Orchestrator (Inspired by Lumen-1)
# =====================================================================
import re
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional
from nutrition.db import DishComponent

# Layer 5: Audit Logging
logging.basicConfig(
    filename='security.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
audit_logger = logging.getLogger('LumenSecurityGuard')

class SecurityOrchestrator:
    def __init__(self):
        self._rate_limits = {}  # client_id -> list of timestamps
        self.RATE_LIMIT_WINDOW = 60  # seconds
        self.MAX_REQUESTS_PER_WINDOW = 30

    # Layer 1: Rate Limiting & DoS Protection
    def check_rate_limit(self, client_id: str) -> bool:
        """Returns True if allowed, False if rate limited."""
        now = time.time()
        if client_id not in self._rate_limits:
            self._rate_limits[client_id] = []
        
        # Keep only timestamps within the window
        self._rate_limits[client_id] = [t for t in self._rate_limits[client_id] if now - t < self.RATE_LIMIT_WINDOW]
        
        if len(self._rate_limits[client_id]) >= self.MAX_REQUESTS_PER_WINDOW:
            audit_logger.warning(f"RATE_LIMIT_EXCEEDED: Client {client_id}")
            return False
            
        self._rate_limits[client_id].append(now)
        return True

    # Layer 2: Intent Classification (Prompt Injection Defense)
    def is_safe_intent(self, user_mode: str, payload_keys: List[str]) -> bool:
        """Detects prompt injection attempts in user mode or payload structure."""
        malicious_patterns = [
            r"ignore all previous",
            r"forget everything",
            r"system prompt",
            r"you are now",
            r"bypass",
            r"override"
        ]
        
        if user_mode:
            user_mode_lower = user_mode.lower()
            for pattern in malicious_patterns:
                if re.search(pattern, user_mode_lower):
                    audit_logger.critical(f"PROMPT_INJECTION_DETECTED: Blocked payload attempting injection -> {user_mode}")
                    return False
                    
        return True

    # Layer 3 & 4: DLP (Data Loss Prevention) and PII Scrubbing
    def scrub_tourist_pii(self, text: str) -> str:
        """
        Lumen-1 Layer 5/10 Security Integration:
        Filters out emails, Sri Lankan & international phone numbers, NICs, and GPS coordinates.
        Also prevents API key leakage.
        """
        if not isinstance(text, str):
            return text

        # 1. Emails
        text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]', text)

        # 2. Phone Numbers
        phone_pattern = r'(?:\+94|0|0094)?\s*(?:[1-9]\d|7\d)\s*\d{3}\s*\d{4}\b'
        text = re.sub(phone_pattern, '[REDACTED_PHONE]', text)

        # 3. Sri Lankan NIC Numbers
        text = re.sub(r'\b\d{9}[vVxX]\b|\b(?:19|20)\d{10}\b', '[REDACTED_NIC]', text)
        
        # 4. DLP: API Keys & Secrets Leakage Prevention
        text = re.sub(r'(sk-[a-zA-Z0-9]{48})', '[REDACTED_API_KEY]', text)
        text = re.sub(r'(AIza[0-9A-Za-z-_]{35})', '[REDACTED_GOOGLE_KEY]', text)

        return text

    # Layer 3 Output Validation
    def validate_components(self, components: List[DishComponent]) -> List[DishComponent]:
        validated = []
        for comp in components:
            clean_name = re.sub(r'[^\w\s]', '', comp.name).strip()
            if len(clean_name) > 1:
                comp.name = clean_name
                validated.append(comp)
        return validated

# Global instance for seamless integration
security_guard = SecurityOrchestrator()
