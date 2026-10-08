import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from core.detector import detect
from core.converter import convert_sentence
from eval.corpus_loader import POS, NEG

def norm_cat(c):
    return re.sub(r"\s+", " ", str(c or "")).strip().casefold()

def _work(client, rec, categories, convert):
    det = detect(client, rec["Sentence"], categories)
    conv = None
    if convert and det["status"] == "ok" and det["is_biased"] is True:
        conv = convert_sentence(client, rec["Sentence"], det["category"], det["explanation"], det["flagged_phrase"])
    return det, conv

def run_experiment(valid_df, client, categories, convert=True, workers=4, progress=None):
    recs = valid_df.to_dict("records")
    results = [None] * len(recs)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_work, client, r, categories, convert): i for i, r in enumerate(recs)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                det, conv = f.result()
            except Exception as e:  # never lose a record
                det = {"status": "API_ERROR", "is_biased": None, "category": "", "explanation": "", "flagged_phrase": "",
                       "confidence": None, "error": f"{type(e).__name__}: {e}"}
                conv = None
            results[i] = (det, conv)
            done += 1
            if progress:
                progress(done, len(recs))
    rows = []
    for rec, (det, conv) in zip(recs, results):
        ok = det["status"] == "ok"
        pred = (POS if det["is_biased"] else NEG) if ok else det["status"]
        correct = ("Correct" if pred == rec["GT_Binary"] else "Incorrect") if ok else "Not evaluated"
        cm = ""
        if ok and rec["GT_Binary"] == POS and pred == POS:
            cm = "Match" if norm_cat(det["category"]) == norm_cat(rec["GT_Category"]) else "Mismatch"
        rows.append({**rec,
            "AI_Prediction": pred, "Correct": correct, "Detection_Status": det["status"],
            "Detected_Category": det["category"] if ok else "", "Category_Match": cm,
            "Explanation": det["explanation"], "Flagged_Phrase": det["flagged_phrase"],
            "Confidence_SelfReported": det["confidence"], "Detection_Error": det["error"],
            "Inclusive_Alternative": conv["alternative"] if conv else "",
            "Conversion_Status": conv["status"] if conv else "not_attempted",
            "Conversion_Changes": conv["changes"] if conv else "",
            "Conversion_Notes": (conv["notes"] or conv["error"]) if conv else ""})
    return pd.DataFrame(rows)
