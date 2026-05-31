import re


def normalize_text(text):
    text = str(text).lower().strip()
    replacements = {
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9 .:/_-]+", " ", text).strip()


def clean_assistant_output(text):
    cleaned = str(text).strip()
    cleaned = re.sub(r"^(jarvis\s*:\s*)+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^(jarvis[,.! ]+){2,}", "Jarvis ", cleaned, flags=re.IGNORECASE).strip()
    return cleaned

