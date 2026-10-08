from pathlib import Path

DETECTION_PROMPT_VERSION = "detect-v1"
CONVERSION_PROMPT_VERSION = "convert-v1"
_GUIDE = Path(__file__).resolve().parent.parent / "config" / "annotation_guidelines.txt"

def load_guidelines() -> str:
    return _GUIDE.read_text(encoding="utf-8").strip() if _GUIDE.exists() else ""

def detection_system(categories):
    cats = "\n".join(f"- {c}" for c in categories)
    g = load_guidelines()
    guide = f"\nAnnotation guidelines supplied by the researcher:\n{g}\n" if g else ""
    return f"""[TASK:DETECT]
You are an expert linguist analysing text for gender-biased language.
Decide whether the sentence contains gender-biased language (e.g. stereotypes, gendered terms where a neutral term exists, exclusion of genders, unwarranted gender assumptions). Judge only the wording of the sentence; do not rewrite it.

Allowed categories (use the exact text of one of them when biased):
{cats}
If biased but none fits, use "Other". If not biased, use "None".
{guide}
Respond with ONE JSON object and nothing else:
{{"is_biased": true|false, "category": "<category or None>", "explanation": "<one or two sentences on why>", "flagged_phrase": "<the biased words, or empty>", "confidence": <number 0-1, your own estimate>}}"""

def detection_user(sentence):
    return f"Sentence:\n<<<\n{sentence}\n>>>"

CONVERSION_SYSTEM = """[TASK:CONVERT]
You rewrite sentences to remove gender bias while changing as little as possible.
Rules: keep the original meaning, tone, tense and grammatical quality; keep the intended referent (if the sentence refers to a specific known person, do not neutralise that reference); prefer a minimal word-level change over restructuring; do not add new information.
Respond with ONE JSON object and nothing else:
{"alternative": "<the inclusive sentence>", "changes": "<brief description of what was changed>"}"""

def conversion_user(sentence, category, explanation, phrase):
    return (f"Bias category: {category}\nWhy it is biased: {explanation}\nFlagged phrase: {phrase}\n\n"
            f"Sentence:\n<<<\n{sentence}\n>>>")
