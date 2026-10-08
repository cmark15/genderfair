import pandas as pd
from eval.corpus_loader import POS, NEG

def safe_div(n, d):
    return n / d if d else None

def f1_of(p, r):
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)

def evaluated(df):
    return df[df.Detection_Status == "ok"]

def binary_metrics(df):
    ev = evaluated(df)
    tp = int(((ev.GT_Binary == POS) & (ev.AI_Prediction == POS)).sum())
    tn = int(((ev.GT_Binary == NEG) & (ev.AI_Prediction == NEG)).sum())
    fp = int(((ev.GT_Binary == NEG) & (ev.AI_Prediction == POS)).sum())
    fn = int(((ev.GT_Binary == POS) & (ev.AI_Prediction == NEG)).sum())
    acc, prec, rec = safe_div(tp + tn, tp + tn + fp + fn), safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    f1 = f1_of(prec, rec)
    notes = []
    if tp + tn + fp + fn == 0: notes.append("Accuracy undefined: no evaluated records.")
    if tp + fp == 0: notes.append("Precision undefined: TP+FP = 0 (no sentence predicted Biased).")
    if tp + fn == 0: notes.append("Recall undefined: TP+FN = 0 (no Biased sentences evaluated).")
    if f1 is None and prec is not None and rec is not None: notes.append("F1 undefined: precision + recall = 0.")
    elif f1 is None: notes.append("F1 undefined because precision and/or recall is undefined.")
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
            "N_evaluated": len(ev), "N_total": len(df), "N_not_evaluated(API/parse errors)": len(df) - len(ev), "Notes": notes}

def confusion_table(df):
    m = binary_metrics(df)
    return pd.DataFrame({"Predicted: Biased": [m["TP"], m["FP"]], "Predicted: Not biased": [m["FN"], m["TN"]]},
                        index=["Actual: Biased", "Actual: Not biased"])

def classification_report(df):
    m = binary_metrics(df)
    tp, tn, fp, fn = m["TP"], m["TN"], m["FP"], m["FN"]
    p1, r1 = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    p0, r0 = safe_div(tn, tn + fn), safe_div(tn, tn + fp)
    rows = [("Biased", p1, r1, f1_of(p1, r1), tp + fn), ("Not biased", p0, r0, f1_of(p0, r0), tn + fp)]
    n = tp + tn + fp + fn
    def macro(i):
        v = [x[i] for x in rows]
        return None if any(a is None for a in v) else sum(v) / 2
    rows.append(("Macro avg", macro(1), macro(2), macro(3), n))
    wa = lambda i: None if any(x[i] is None for x in rows[:2]) or n == 0 else sum(x[i] * x[4] for x in rows[:2]) / n
    rows.append(("Weighted avg", wa(1), wa(2), wa(3), n))
    rows.append(("Accuracy", None, None, m["Accuracy"], n))
    return pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1", "Support"])

def category_analysis(df):
    ev = evaluated(df)
    out = []
    cats = sorted(c for c in df[df.GT_Binary == POS].GT_Category.unique() if c)
    for c in cats:
        allc = df[(df.GT_Binary == POS) & (df.GT_Category == c)]
        e = ev[(ev.GT_Binary == POS) & (ev.GT_Category == c)]
        det = e[e.AI_Prediction == POS]
        cm = int((det.Category_Match == "Match").sum())
        from eval.experiment import norm_cat
        pred_c = ev[(ev.AI_Prediction == POS) & (ev.Detected_Category.map(norm_cat) == norm_cat(c))]
        out.append({"Category": c, "Samples": len(allc), "Evaluated": len(e), "Not_evaluated": len(allc) - len(e),
                    "Detected_as_biased(TP)": len(det), "Missed(FN)": len(e) - len(det),
                    "Binary_recall": safe_div(len(det), len(e)),
                    "Category_correct": cm, "Category_incorrect(of_detected)": len(det) - cm,
                    "Category_accuracy(of_evaluated)": safe_div(cm, len(e)),
                    "Category_accuracy(of_detected)": safe_div(cm, len(det)),
                    "Predicted_as_category": len(pred_c),
                    "Category_precision": safe_div(int((pred_c.GT_Category.map(norm_cat) == norm_cat(c)).sum()), len(pred_c))})
    nb = df[df.GT_Binary == NEG]; nbe = ev[ev.GT_Binary == NEG]
    out.append({"Category": "(Not biased - no category)", "Samples": len(nb), "Evaluated": len(nbe), "Not_evaluated": len(nb) - len(nbe),
                "Detected_as_biased(TP)": None, "Missed(FN)": None,
                "Binary_recall": safe_div(int((nbe.AI_Prediction == NEG).sum()), len(nbe)),  # = specificity
                "Category_correct": None, "Category_incorrect(of_detected)": None,
                "Category_accuracy(of_evaluated)": None, "Category_accuracy(of_detected)": None,
                "Predicted_as_category": None, "Category_precision": None})
    return pd.DataFrame(out)

def error_analysis(df, low_conf=0.7, conflict_sentences=()):
    cs = {s.casefold() for s in conflict_sentences}
    rows = []
    for _, r in df.iterrows():
        if r.Detection_Status != "ok":
            rows.append((r, "API/parse error (not evaluated)", f"{r.Detection_Status}: {r.Detection_Error}")); continue
        flags = []
        if r.Confidence_SelfReported is not None and pd.notna(r.Confidence_SelfReported) and r.Confidence_SelfReported < low_conf:
            flags.append(f"low self-reported confidence (<{low_conf})")
        if r.Sentence.casefold() in cs:
            flags.append("duplicate sentence with conflicting ground-truth labels")
        if r.Correct == "Incorrect":
            et = "False positive" if r.AI_Prediction == POS else "False negative"
            rows.append((r, et, r.Explanation, flags))
        elif r.Category_Match == "Mismatch":
            rows.append((r, "Category mismatch (binary correct)", f"GT category '{r.GT_Category}' vs predicted '{r.Detected_Category}'. {r.Explanation}", flags))
        elif flags:
            rows.append((r, "Correct but flagged as potentially difficult", r.Explanation, flags))
    out = []
    for t in rows:
        r, et, ex = t[0], t[1], t[2]
        fl = t[3] if len(t) > 3 else []
        out.append({"ID": r.ID, "Sentence": r.Sentence, "Ground Truth": r.Ground_Truth, "AI Prediction": r.AI_Prediction,
                    "Error Type": et, "Explanation": ex, "GT_Category": r.GT_Category, "AI_Category": r.Detected_Category,
                    "Difficulty_Flags": "; ".join(fl)})
    cols = ["ID", "Sentence", "Ground Truth", "AI Prediction", "Error Type", "Explanation", "GT_Category", "AI_Category", "Difficulty_Flags"]
    return pd.DataFrame(out, columns=cols)

def error_patterns(err):
    """Descriptive counts only; qualitative interpretation is left to the researcher."""
    e = err[err["Error Type"].isin(["False positive", "False negative"])]
    pats = {}
    pats["by_error_type"] = e["Error Type"].value_counts().to_dict()
    pats["false_negatives_by_gt_category"] = e[e["Error Type"] == "False negative"].GT_Category.value_counts().to_dict()
    pats["false_positives_by_ai_category"] = e[e["Error Type"] == "False positive"].AI_Category.value_counts().to_dict()
    return pats

def conversion_summary(df):
    cand = df[df.AI_Prediction == POS]
    ok = cand[cand.Conversion_Status == "ok"]
    unchanged = ok[ok.Conversion_Notes.str.contains("unchanged", na=False)]
    problem = cand[(cand.Conversion_Status != "ok") | cand.Conversion_Notes.str.contains("unchanged|much_", na=False)]
    return {"AI-classified biased (conversion candidates)": len(cand),
            "Conversion returned a valid alternative": len(ok),
            "  of which identical to original (unchanged)": len(unchanged),
            "Conversion API/parse failures": int(cand.Conversion_Status.isin(["API_ERROR", "PARSE_ERROR"]).sum()),
            "Candidates whose ground truth is Not biased (false positives)": int((cand.GT_Binary == NEG).sum()),
            "Flagged problematic outputs (failed/unchanged/length anomaly)": len(problem)}, problem
