"""Builds every table from actual data. Nothing here invents values."""
import pandas as pd
from eval import metrics as M
from eval.corpus_loader import POS, NEG
from eval.expert_stats import expert_summary

def fmt(x, nd=4):
    return "undefined" if x is None or (isinstance(x, float) and pd.isna(x)) else (round(x, nd) if isinstance(x, float) else x)

def compute_all(results, quality, ratings=None, instrument=None):
    bm = M.binary_metrics(results)
    conflict_sents = [g["Sentence"] for g in quality.get("conflicting_duplicate_groups", [])]
    err = M.error_analysis(results, conflict_sentences=conflict_sents)
    conv_sum, conv_problem = M.conversion_summary(results)
    exp = expert_summary(ratings, instrument) if (ratings is not None and instrument) else None
    T = {}
    cats = quality.get("category_counts", {})
    nb = quality["n_biased"]
    t1 = [{"Group": c, "n": n, "% of valid corpus": 100 * n / max(quality["n_valid"], 1)} for c, n in cats.items()]
    if quality.get("biased_rows_without_category"):
        t1.append({"Group": "(Biased, category missing)", "n": quality["biased_rows_without_category"], "% of valid corpus": 100 * quality["biased_rows_without_category"] / max(quality["n_valid"], 1)})
    t1 += [{"Group": "Total biased", "n": nb, "% of valid corpus": 100 * nb / max(quality["n_valid"], 1)},
           {"Group": "Total not biased", "n": quality["n_not_biased"], "% of valid corpus": 100 * quality["n_not_biased"] / max(quality["n_valid"], 1)},
           {"Group": "Total valid", "n": quality["n_valid"], "% of valid corpus": 100.0}]
    T["Table 1. Corpus Distribution"] = pd.DataFrame(t1)
    T["Table 2. Gender-Bias Detection Results"] = pd.DataFrame(
        [{"Metric": k, "Value": fmt(bm[k])} for k in ["TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1"]]
        + [{"Metric": "N evaluated", "Value": bm["N_evaluated"]}, {"Metric": "N not evaluated (API/parse errors)", "Value": bm["N_not_evaluated(API/parse errors)"]}])
    T["Table 3. Confusion Matrix"] = M.confusion_table(results).reset_index().rename(columns={"index": ""})
    ca = M.category_analysis(results)
    T["Table 4. Performance by Bias Category"] = ca.map(fmt) if hasattr(ca, "map") else ca.applymap(fmt)
    T["Table 5. Error Analysis"] = err[["ID", "Sentence", "Ground Truth", "AI Prediction", "Error Type", "Explanation"]]
    ex = results[results.Conversion_Status == "ok"].head(10)
    T["Table 6. Sample Detection and Conversion Results"] = ex[["ID", "Sentence", "AI_Prediction", "Detected_Category", "Inclusive_Alternative"]].rename(
        columns={"Sentence": "Original Sentence", "AI_Prediction": "Detected Bias", "Detected_Category": "Bias Category", "Inclusive_Alternative": "Inclusive Alternative"})
    if exp is not None:
        c = exp["criterion"].copy(); c.insert(0, "", "")
        T["Table 7. Expert Validation Results"] = exp["criterion"]
    return {"binary": bm, "confusion": M.confusion_table(results), "report": M.classification_report(results),
            "category": ca, "errors": err, "patterns": M.error_patterns(err), "conversion_summary": conv_sum,
            "conversion_problem": conv_problem, "expert": exp, "tables": T}

def results_draft(run, A):
    bm, q, mf = A["binary"], run["quality"], run["manifest"]
    L = []
    if mf.get("is_mock"):
        L += ["> **WARNING: this run used the mock test provider. Do not report any of these numbers.**\n"]
    L += ["# Results and Discussion (auto-drafted from actual run data; researcher must review and interpret)\n",
          f"Run: `{run['dir'].name}` | Model requested: `{mf['model_requested']}` (returned: `{mf.get('model_returned_by_api')}`) | Temperature requested: {mf['temperature_requested']} (applied: {mf['temperature_applied']}) | Started (UTC): {mf['experiment_started_utc']}\n",
          "## Corpus", f"The workbook contained {q['n_rows_total']} data rows; {q['n_valid']} were valid and evaluable ({q['n_biased']} labelled biased, {q['n_not_biased']} not biased) and {q['n_excluded']} were excluded (missing sentence: {q['missing_sentence']}, missing label: {q['missing_label']}, unmapped label: {q['unmapped_label']}). {q['duplicate_sentence_rows']} rows shared a sentence with another row; {len(q['conflicting_duplicate_groups'])} duplicate group(s) had conflicting labels.\n",
          "## Detection (Objective 1)",
          f"Of {bm['N_total']} processed sentences, {bm['N_evaluated']} were scored and {bm['N_not_evaluated(API/parse errors)']} could not be scored because of API/parse errors. TP={bm['TP']}, TN={bm['TN']}, FP={bm['FP']}, FN={bm['FN']}. Accuracy={fmt(bm['Accuracy'])}, precision={fmt(bm['Precision'])}, recall={fmt(bm['Recall'])}, F1={fmt(bm['F1'])}."]
    L += [f"- Note: {n}" for n in bm["Notes"]]
    p = A["patterns"]
    L += [f"\nError counts: {p['by_error_type']}. False negatives by ground-truth category: {p['false_negatives_by_gt_category']}. False positives by AI category: {p['false_positives_by_ai_category']}. (Descriptive counts only; qualitative review of Table 5 is required before drawing conclusions about error causes.)\n",
          "## Conversion (Objective 2)"]
    L += [f"- {k}: {v}" for k, v in A["conversion_summary"].items()]
    L += ["\nThese counts describe whether a rewrite was produced; they do **not** measure its quality. Quality is assessed by expert validation.\n", "## Expert validation (Objective 3)"]
    e = A["expert"]
    if e is None:
        L += ["No expert ratings were available for this run, so no expert-validation results are reported."]
    else:
        o = e["overall"]
        L += [f"{o['Raters']} rater(s) provided {o['Total_ratings']} ratings on {o['Items_rated']} item(s). Overall mean of criterion means: {fmt(o['Overall_mean(of_criterion_means)'])}.", e["criterion"].to_markdown(index=False) if _has_tabulate() else e["criterion"].to_string(index=False)]
        if len(e["agreement"]):
            L += ["Agreement (only items rated by 2+ raters):", e["agreement"].to_string(index=False)]
        else:
            L += ["Inter-rater agreement not computed (no item was rated by 2+ raters)."]
    L += ["\n## Limitations to state in the paper",
          "- Confidence values are self-reported by the LLM and are not calibrated probabilities.",
          "- Results are specific to the model, prompts and temperature recorded in manifest.json.",
          "- Whether the system is 'effective' or 'reliable' is a judgement for the researcher based on the figures above and the corpus size; this draft makes no such claim."]
    return "\n".join(L)

def _has_tabulate():
    try:
        import tabulate  # noqa
        return True
    except Exception:
        return False
