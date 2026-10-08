"""Preprocessing is deliberately non-destructive: whitespace/Unicode normalisation only.
No lower-casing, stop-word removal or punctuation stripping, so the detector sees the wording as written."""
import re, unicodedata

def normalize(text) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text)).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", t).strip()

def split_sentences(text: str):
    t = normalize(text)
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", t)
    return [p.strip() for p in parts if p.strip()]
