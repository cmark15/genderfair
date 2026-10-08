from .llm_client import LLMError
from .detector import extract_json
from .prompts import CONVERSION_SYSTEM, conversion_user

def convert_sentence(client, sentence, category, explanation, phrase):
    out = {"status": "ok", "alternative": "", "changes": "", "notes": "", "error": ""}
    try:
        raw = client.complete(CONVERSION_SYSTEM, conversion_user(sentence, category, explanation, phrase))
    except LLMError as e:
        out.update(status="API_ERROR", error=str(e))
        return out
    try:
        d = extract_json(raw)
        alt = str(d.get("alternative", "")).strip()
        out["changes"] = str(d.get("changes", "")).strip()
        if not alt:
            out["notes"] = "empty_alternative"
            out["status"] = "PARSE_ERROR"
            return out
        out["alternative"] = alt
        notes = []
        if alt.strip().casefold() == sentence.strip().casefold():
            notes.append("unchanged")
        ratio = len(alt) / max(len(sentence), 1)
        if ratio > 1.5:
            notes.append("much_longer(>1.5x)")
        if ratio < 0.5:
            notes.append("much_shorter(<0.5x)")
        out["notes"] = "; ".join(notes)
    except Exception as e:  # noqa
        out.update(status="PARSE_ERROR", error=f"{type(e).__name__}: {e}")
    return out
