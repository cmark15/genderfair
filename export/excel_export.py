import pandas as pd
from pathlib import Path

def _kv(d):
    return pd.DataFrame([{"Item": k, "Value": str(v) if isinstance(v, (dict, list)) else v} for k, v in d.items()])

def write_workbook(path, run, A):
    q, mf, res = run["quality"], run["manifest"], run["results"]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        r = 0
        summ = {**{k: v for k, v in q.items() if k not in ("conflicting_duplicate_groups", "label_counts", "category_counts", "label_variants", "category_spelling_variants")},
                "label_counts": q["label_counts"], "category_counts": q["category_counts"], "label_variants": q["label_variants"],
                "category_spelling_variants": q["category_spelling_variants"], "conflicting_duplicate_groups": q["conflicting_duplicate_groups"]}
        if mf.get("is_mock"):
            summ = {"WARNING": "MOCK PROVIDER RUN - NOT VALID RESEARCH RESULTS", **summ}
        s = _kv(summ); s.to_excel(xw, sheet_name="Corpus Summary", index=False)
        run["excluded"].to_excel(xw, sheet_name="Corpus Summary", index=False, startrow=len(s) + 3)
        res.to_excel(xw, sheet_name="Sentence-Level Results", index=False)
        A["confusion"].to_excel(xw, sheet_name="Confusion Matrix")
        m = _kv({k: ("undefined" if v is None else v) for k, v in A["binary"].items() if k != "Notes"})
        m.to_excel(xw, sheet_name="Classification Metrics", index=False)
        A["report"].to_excel(xw, sheet_name="Classification Metrics", index=False, startrow=len(m) + 3)
        pd.DataFrame({"Notes": A["binary"]["Notes"] or ["(none)"]}).to_excel(xw, sheet_name="Classification Metrics", index=False, startrow=len(m) + 14)
        A["category"].to_excel(xw, sheet_name="Category Analysis", index=False)
        A["errors"].to_excel(xw, sheet_name="Error Analysis", index=False)
        conv = res[res.AI_Prediction == "Biased"][["ID", "Sentence", "GT_Binary", "AI_Prediction", "Detected_Category", "Explanation", "Inclusive_Alternative", "Conversion_Status", "Conversion_Changes", "Conversion_Notes"]]
        conv.rename(columns={"Sentence": "Original Sentence", "Detected_Category": "Bias Category", "Inclusive_Alternative": "Inclusive Alternative"}).to_excel(xw, sheet_name="Conversion Results", index=False)
        e = A["expert"]
        if e is None:
            pd.DataFrame({"Note": ["No expert-validation data available for this run."]}).to_excel(xw, sheet_name="Expert Validation", index=False)
        else:
            row = 0
            for title, df in [("Criterion-level", e["criterion"]), ("Frequencies", e["frequency"]), ("Agreement", e["agreement"]), ("Overall", _kv(e["overall"])), ("Comments", e["comments"])]:
                pd.DataFrame({title: []}).to_excel(xw, sheet_name="Expert Validation", index=False, startrow=row)
                df.to_excel(xw, sheet_name="Expert Validation", index=False, startrow=row + 1); row += len(df) + 4
        row = 0
        for title, df in A["tables"].items():
            pd.DataFrame({title: []}).to_excel(xw, sheet_name="Research Tables", index=False, startrow=row)
            df.to_excel(xw, sheet_name="Research Tables", index=False, startrow=row + 1); row += len(df) + 4
        for ws in xw.book.worksheets:
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 60)
    return Path(path)
