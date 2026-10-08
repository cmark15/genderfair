"""Read-only corpus inspection. The uploaded file is never written to; everything operates on in-memory copies."""
import io, re
import pandas as pd
from core.preprocess import normalize

POS, NEG = "Biased", "Not biased"

def read_workbook(file_like, filename="corpus.xlsx"):
    data = file_like.read() if hasattr(file_like, "read") else file_like
    bio = io.BytesIO(data)
    if filename.lower().endswith(".csv"):
        return {"(csv)": pd.read_csv(bio, dtype=object)}
    return pd.read_excel(bio, sheet_name=None, dtype=object)

def _isblank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "" or str(v).strip().lower() in ("nan", "none_value")

def infer_columns(df):
    """Heuristic SUGGESTIONS only; the user confirms in the UI."""
    cols = list(df.columns)
    low = {c: str(c).strip().casefold() for c in cols}
    def find(pats, exclude=()):
        for c in cols:
            if c in exclude:
                continue
            if any(re.search(p, low[c]) for p in pats):
                return c
        return None
    sent = find([r"sentence", r"text", r"statement", r"utterance"])
    if sent is None:
        lens = {c: df[c].dropna().astype(str).str.len().mean() for c in cols}
        lens = {c: v for c, v in lens.items() if v == v}
        sent = max(lens, key=lens.get) if lens else None
    idc = find([r"^id$", r"\bid\b", r"^no\.?$", r"^#$", r"number"], exclude=(sent,))
    cat = find([r"categor", r"type", r"class"], exclude=(sent,))
    lab = find([r"ground", r"truth", r"label", r"biased", r"annot", r"gold"], exclude=(sent, cat))
    return {"sentence": sent, "id": idc, "label": lab or cat, "category": cat}

def suggest_label_sides(values):
    pos, neg, unk = [], [], []
    for v in values:
        t = str(v).strip().casefold()
        if re.match(r"^(not|non|no\b|un|neutral|0$|false|fair|inclusive|n$)", t):
            neg.append(v)
        elif "bias" in t or t in ("1", "yes", "true", "y"):
            pos.append(v)
        else:
            unk.append(v)
    return pos, neg, unk

def prepare_corpus(df, mapping, pos_values, neg_values):
    """Returns (valid_df, excluded_df, quality). Ground-truth labels are mapped, never changed."""
    sc, ic, lc, cc = mapping["sentence"], mapping.get("id"), mapping["label"], mapping.get("category")
    pos = {str(v).strip() for v in pos_values}
    neg = {str(v).strip() for v in neg_values}
    rows, excluded = [], []
    for i, r in df.iterrows():
        src = i + 2  # spreadsheet row, assuming header on row 1
        s_raw, l_raw = r[sc], r[lc]
        if _isblank(s_raw):
            excluded.append({"Source_Row": src, "ID": r[ic] if ic else "", "Reason": "Missing sentence"}); continue
        if _isblank(l_raw):
            excluded.append({"Source_Row": src, "ID": r[ic] if ic else "", "Sentence": normalize(s_raw), "Reason": "Missing ground-truth label"}); continue
        lt = str(l_raw).strip()
        if lt in pos:
            gb = POS
        elif lt in neg:
            gb = NEG
        else:
            excluded.append({"Source_Row": src, "ID": r[ic] if ic else "", "Sentence": normalize(s_raw), "Reason": f"Unmapped/inconsistent label: '{lt}'"}); continue
        gcat = ""
        if gb == POS:
            gcat = normalize(r[cc]) if cc and not _isblank(r[cc]) else ""
        rows.append({"Source_Row": src, "ID": (r[ic] if ic and not _isblank(r[ic]) else src), "Sentence": normalize(s_raw),
                     "Ground_Truth": lt, "GT_Binary": gb, "GT_Category": gcat})
    valid = pd.DataFrame(rows, columns=["Source_Row", "ID", "Sentence", "Ground_Truth", "GT_Binary", "GT_Category"])
    excl = pd.DataFrame(excluded)

    q = {"n_rows_total": int(len(df)), "n_valid": int(len(valid)), "n_excluded": int(len(excl)),
         "n_biased": int((valid.GT_Binary == POS).sum()), "n_not_biased": int((valid.GT_Binary == NEG).sum()),
         "id_generated": ic is None}
    q["missing_sentence"] = int((excl.Reason == "Missing sentence").sum()) if len(excl) else 0
    q["missing_label"] = int((excl.Reason == "Missing ground-truth label").sum()) if len(excl) else 0
    q["unmapped_label"] = int(excl.Reason.str.startswith("Unmapped").sum()) if len(excl) else 0
    # duplicates (kept in the experiment, only reported)
    key = valid.Sentence.str.casefold()
    dup = valid[key.duplicated(keep=False)].copy()
    q["duplicate_sentence_rows"] = int(len(dup))
    conflicts = []
    for _, g in dup.groupby(dup.Sentence.str.casefold()):
        if g.GT_Binary.nunique() > 1:
            conflicts.append({"Sentence": g.Sentence.iloc[0], "Source_Rows": list(g.Source_Row), "Labels": list(g.Ground_Truth)})
    q["conflicting_duplicate_groups"] = conflicts
    q["duplicate_ids"] = int(valid.ID.astype(str).duplicated(keep=False).sum())
    # label formatting variants
    raw_vals = df[lc].dropna().astype(str)
    stripped = raw_vals.str.strip()
    variants = {}
    for v in stripped.unique():
        variants.setdefault(v.casefold(), set()).add(v)
    q["label_variants"] = {k: sorted(v) for k, v in variants.items() if len(v) > 1}
    q["whitespace_padded_labels"] = int((raw_vals != stripped).sum())
    q["label_counts"] = stripped.value_counts().to_dict()
    cats = valid[valid.GT_Binary == POS].GT_Category
    q["biased_rows_without_category"] = int((cats == "").sum())
    q["category_counts"] = cats[cats != ""].value_counts().to_dict()
    vv = {}
    for c in q["category_counts"]:
        vv.setdefault(re.sub(r"\s+", " ", c).casefold(), []).append(c)
    q["category_spelling_variants"] = {k: v for k, v in vv.items() if len(v) > 1}
    return valid, excl, q
