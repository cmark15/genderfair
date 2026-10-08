import json
from pathlib import Path
import numpy as np
import pandas as pd

def load_instrument(path_or_dict=None):
    if isinstance(path_or_dict, dict):
        return path_or_dict
    p = Path(path_or_dict) if path_or_dict else Path(__file__).resolve().parent.parent / "config" / "expert_instrument.json"
    return json.loads(p.read_text(encoding="utf-8"))

def krippendorff_alpha_interval(units):
    """units: list of lists of ratings (one list per item). Interval metric; only items with >=2 ratings count."""
    u = [np.array(v, float) for v in units if len(v) >= 2]
    if not u:
        return None
    allv = np.concatenate(u); n = len(allv)
    if n < 2:
        return None
    de = 2 * (n * (allv ** 2).sum() - allv.sum() ** 2) / (n * (n - 1))
    do = sum(2 * (len(v) * (v ** 2).sum() - v.sum() ** 2) / (len(v) - 1) for v in u) / n
    return None if de == 0 else 1 - do / de

def expert_summary(ratings, instrument):
    """ratings: long df [item_key, rater_id, criterion, rating, comment]. Returns dict of tables or None."""
    if ratings is None or len(ratings) == 0:
        return None
    lo, hi = int(instrument["scale_min"]), int(instrument["scale_max"])
    scale = list(range(lo, hi + 1))
    labels = {c["key"]: c["label"] for c in instrument["criteria"]}
    r = ratings.copy(); r["rating"] = pd.to_numeric(r["rating"], errors="coerce")
    r = r.dropna(subset=["rating"])
    rows, freq_rows, agree_rows = [], [], []
    bands = instrument.get("interpretation_bands")
    def band(m):
        if not bands: return ""
        for b in bands:
            if b["min"] <= m <= b["max"]: return b["label"]
        return ""
    for key, lab in labels.items():
        x = r[r.criterion == key].rating
        if len(x) == 0: continue
        f = x.value_counts().reindex(scale, fill_value=0)
        wm = float((np.array(f.index) * f.values).sum() / f.sum())
        rows.append({"Criterion": lab, "n_ratings": int(len(x)), "Mean": float(x.mean()), "Weighted_mean(freq)": wm,
                     "SD(sample)": float(x.std(ddof=1)) if len(x) > 1 else None, "Interpretation": band(float(x.mean()))})
        for s in scale:
            freq_rows.append({"Criterion": lab, "Rating": s, "Frequency": int(f[s]), "Percent": float(f[s] / f.sum() * 100)})
        g = r[r.criterion == key].groupby("item_key").rating.apply(list)
        multi = [v for v in g if len(v) >= 2]
        if multi:
            pair_exact = pair_adj = pairs = 0
            for v in multi:
                for i in range(len(v)):
                    for j in range(i + 1, len(v)):
                        pairs += 1; pair_exact += v[i] == v[j]; pair_adj += abs(v[i] - v[j]) <= 1
            agree_rows.append({"Criterion": lab, "Items_with_2+_raters": len(multi), "Pairs": pairs,
                               "Exact_agreement_%": pair_exact / pairs * 100, "Within_1_point_%": pair_adj / pairs * 100,
                               "Krippendorff_alpha(interval)": krippendorff_alpha_interval(multi)})
    crit = pd.DataFrame(rows)
    overall = {"Overall_mean(of_criterion_means)": float(crit.Mean.mean()) if len(crit) else None,
               "Raters": int(r.rater_id.nunique()), "Items_rated": int(r.item_key.nunique()), "Total_ratings": int(len(r))}
    if overall["Overall_mean(of_criterion_means)"] is not None:
        overall["Overall_interpretation"] = band(overall["Overall_mean(of_criterion_means)"])
    comments = ratings[ratings.get("comment", pd.Series(dtype=object)).fillna("").astype(str).str.strip() != ""]
    cm = comments[["item_key", "rater_id", "comment"]].drop_duplicates() if len(comments) else pd.DataFrame(columns=["item_key", "rater_id", "comment"])
    return {"criterion": crit, "frequency": pd.DataFrame(freq_rows), "agreement": pd.DataFrame(agree_rows),
            "overall": overall, "comments": cm}
