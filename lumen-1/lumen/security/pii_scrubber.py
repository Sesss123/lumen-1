import re
from lumen.security import is_layer_enabled

def scrub_pii(text: str) -> str:
    """
    Scrubs sensitive data from model outputs to prevent Data Leakage (Layer 5/10).
    Filters out emails, local and international phone numbers, Sri Lankan NIC formats,
    Credit cards, and GPS coordinates.
    """
    if not is_layer_enabled(5) and not is_layer_enabled(10):
        return text
        
    if not isinstance(text, str):
        return text

    # 1. Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '[REDACTED_EMAIL]', text)

    # 2. Sri Lankan & International Phone Numbers
    phone_pattern = r'(?:\+94|0|0094)?\s*(?:[1-9]\d|7\d)\s*\d{3}\s*\d{4}\b|(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}\b'
    def phone_repl(match):
        m = match.group(0)
        digits = re.sub(r'\D', '', m)
        if len(digits) >= 8 and len(digits) <= 15:
            return '[REDACTED_PHONE]'
        return m
    text = re.sub(phone_pattern, phone_repl, text)

    # 3. Sri Lankan NIC Numbers (Old: 9 digits + V/X, New: 12 digits starting with 19 or 20)
    old_nic_pattern = r'\b\d{9}[vVxX]\b'
    new_nic_pattern = r'\b(?:19|20)\d{10}\b'
    text = re.sub(old_nic_pattern, '[REDACTED_NIC]', text)
    text = re.sub(new_nic_pattern, '[REDACTED_NIC]', text)

    # 4. Credit Card Patterns (13 to 19 digits, standard spacing/formatting)
    card_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b'
    def card_repl(match):
        c = match.group(0)
        digits = re.sub(r'\D', '', c)
        if len(digits) in (13, 14, 15, 16, 19):
            # Run a quick Luhn check to avoid false positives on years or large integer lists
            total = 0
            reverse_digits = digits[::-1]
            for idx, digit in enumerate(reverse_digits):
                n = int(digit)
                if idx % 2 == 1:
                    n *= 2
                    if n > 9:
                        n -= 9
                total += n
            if total % 10 == 0:
                return '[REDACTED_CARD]'
        return c
    text = re.sub(card_pattern, card_repl, text)

    # 5. GPS Coordinates — Bug 7 Fix: require BOTH lat AND lon, realistic ranges,
    #    and at least 3 decimal digits so training values (0.00035, 6.9) are NOT matched.
    #    Latitude:  -90  to  90   (e.g. 6.927100)
    #    Longitude: -180 to 180   (e.g. 79.861200)
    #    Pattern requires comma separator between the pair.
    gps_pattern = (
        r'\b'
        r'(?P<lat>[-+]?(?:90(?:\.0+)?|[1-8]?\d\.\d{3,}))'   # lat: 3+ decimals
        r'\s*,\s*'
        r'(?P<lon>[-+]?(?:180(?:\.0+)?|1[0-7]\d\.\d{3,}|[1-9]?\d\.\d{3,}))'  # lon: 3+ decimals
        r'\b'
    )
    text = re.sub(gps_pattern, '[REDACTED_GPS]', text)


    return text
