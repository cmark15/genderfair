import json, re
from .llm_client import LLMError
from .prompts import detection_system, detection_user

def extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))

def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no"):
        return False
    raise ValueError(f"is_biased not boolean: {v!r}")

def canonical_category(cat, categories):
    c = re.sub(r"\s+", " ", str(cat or "")).strip().casefold()
    for k in categories:
        if re.sub(r"\s+", " ", k).strip().casefold() == c:
            return k
    return str(cat or "").strip()

def detect(client, sentence, categories):
    out = {"status": "ok", "is_biased": None, "category": "", "explanation": "", "flagged_phrase": "",
           "confidence": None, "error": "", "raw": ""}
    try:
        raw = client.complete(detection_system(categories), detection_user(sentence))
        out["raw"] = raw
    except LLMError as e:
        out.update(status="API_ERROR", error=str(e))
        return out
    try:
        d = extract_json(raw)
        out["is_biased"] = _bool(d.get("is_biased"))
        out["category"] = canonical_category(d.get("category"), categories) if out["is_biased"] else "None"
        out["explanation"] = str(d.get("explanation", "")).strip()
        out["flagged_phrase"] = str(d.get("flagged_phrase", "")).strip()
        try:
            cf = float(d.get("confidence"))
            out["confidence"] = cf if 0 <= cf <= 1 else None
        except (TypeError, ValueError):
            out["confidence"] = None
    except Exception as e:  # noqa
        out.update(status="PARSE_ERROR", error=f"{type(e).__name__}: {e}", is_biased=None)
    return out
