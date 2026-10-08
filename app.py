import os, json, io, zipfile, datetime
from pathlib import Path
import pandas as pd
import streamlit as st
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

from core.llm_client import LLMClient, LLMError, DEFAULT_MODELS
from core.preprocess import split_sentences, normalize
from core.detector import detect
from core.converter import convert_sentence
from core.manifest import build_manifest
from core import runstore
from eval.corpus_loader import read_workbook, infer_columns, suggest_label_sides, prepare_corpus, POS, NEG
from eval.experiment import run_experiment
from eval.expert_stats import load_instrument
from export.report import compute_all, results_draft, fmt
from export.figures import make_figures
from export.excel_export import write_workbook

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Gender-Bias Detection & Inclusive Conversion", layout="wide")
st.title("AI-Driven Gender-Bias Detection and Inclusive-Language Conversion")
st.caption("Research prototype · detection and conversion are separate modules · ground-truth labels are never modified")

# ---------------- sidebar: configuration ----------------
with st.sidebar:
    st.header("LLM configuration")
    prov_opts = ["anthropic", "openai", "mock"]
    provider = st.selectbox("Provider", prov_opts, index=prov_opts.index(os.getenv("LLM_PROVIDER", "anthropic")) if os.getenv("LLM_PROVIDER", "anthropic") in prov_opts else 0)
    model = st.text_input("Model", value=os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider], help="Recorded in the reproducibility manifest.")
    temperature = st.number_input("Temperature", 0.0, 1.0, float(os.getenv("LLM_TEMPERATURE", "0")), 0.1)
    workers = st.slider("Parallel requests", 1, 16, 4)
    client = LLMClient(provider, model, temperature)
    if provider == "mock":
        st.error("MOCK provider: rule-based test double. NEVER use for reported results.")
    elif client.key_present():
        st.success("API key found in environment.")
    else:
        st.warning(f"No API key in environment ({'ANTHROPIC_API_KEY' if provider=='anthropic' else 'OPENAI_API_KEY'}). Set it before running; keys are never entered or stored here.")

def default_categories():
    run = st.session_state.get("run")
    if run:
        return run["manifest"]["categories_given_to_detector"], "corpus"
    return json.loads((ROOT / "config" / "categories.json").read_text())["candidate_categories"], "candidate"

tab1, tab2, tab3, tab4 = st.tabs(["1 · Analyze text", "2 · Corpus experiment", "3 · Results dashboard", "4 · Expert validation"])

# ---------------- tab 1 ----------------
with tab1:
    st.subheader("Analyze a sentence or paragraph")
    cats, src = default_categories()
    if src == "candidate":
        st.info("No corpus loaded: using CANDIDATE categories from config/categories.json. Load a corpus (tab 2) to use the corpus categories.")
    st.session_state.setdefault("input_text", "")
    text = st.text_area("Text", key="input_text", height=140, placeholder="Paste a sentence or paragraph…")
    c1, c2 = st.columns([1, 1])
    go = c1.button("Analyze", type="primary")
    c2.button("Analyze another text", on_click=lambda: st.session_state.update(input_text="", single=None))
    if go:
        if not text.strip():
            st.warning("Enter some text first.")
        elif not client.is_mock and not client.key_present():
            st.error("No API key configured.")
        else:
            sents = split_sentences(text)                       # preprocessing
            rows, final = [], []
            with st.spinner("Analyzing…"):
                for s in sents:
                    d = detect(client, s, cats)                 # detection + classification
                    cv = convert_sentence(client, s, d["category"], d["explanation"], d["flagged_phrase"]) if d["is_biased"] else None  # conversion
                    rows.append((s, d, cv))
                    final.append(cv["alternative"] if cv and cv["status"] == "ok" else s)
            st.session_state["single"] = (rows, " ".join(final))
    if st.session_state.get("single"):
        rows, joined = st.session_state["single"]
        nb = sum(1 for _, d, _ in rows if d["is_biased"])
        st.markdown(f"**Gender-biased language detected in {nb} of {len(rows)} sentence(s).**")
        for s, d, cv in rows:
            with st.container(border=True):
                if d["status"] != "ok":
                    st.error(f"{d['status']}: {d['error']}  \n_{s}_"); continue
                if d["is_biased"]:
                    a, b = st.columns(2)
                    a.markdown("**Original**"); a.write(s)
                    b.markdown("**Suggested inclusive alternative**")
                    b.write(cv["alternative"] if cv and cv["status"] == "ok" else f"(conversion failed: {cv['error'] if cv else ''})")
                    st.markdown(f"**Category:** {d['category']}  ·  **Flagged:** {d['flagged_phrase'] or '—'}" + (f"  ·  **Self-reported confidence:** {d['confidence']}" if d["confidence"] is not None else ""))
                    st.markdown(f"**Why it may be biased:** {d['explanation']}")
                    if cv and cv["changes"]: st.caption(f"Changes: {cv['changes']}")
                else:
                    st.markdown(f"✅ {s}"); st.caption(f"No gender bias detected. {d['explanation']}")
        if nb:
            st.markdown("**Full text with inclusive alternatives applied**"); st.write(joined)

# ---------------- tab 2 ----------------
with tab2:
    st.subheader("Upload corpus workbook")
    st.write("Upload your Excel file (.xlsx/.xls) or .csv. It is read in memory only and never modified; a byte-for-byte copy is stored with each run.")
    up = st.file_uploader("Corpus file", type=["xlsx", "xls", "csv"])
    if up is not None:
        raw = up.getvalue()
        try:
            sheets = read_workbook(io.BytesIO(raw), up.name)
        except Exception as e:
            st.error(f"Could not read file: {e}"); sheets = {}
        if sheets:
            st.markdown("**Workbook inspection**")
            st.dataframe(pd.DataFrame([{"Sheet": n, "Rows": len(d), "Columns": ", ".join(map(str, d.columns))} for n, d in sheets.items()]))
            sheet = st.selectbox("Sheet containing the test corpus", list(sheets))
            df = sheets[sheet].reset_index(drop=True)
            st.dataframe(df.head(10).astype(str))
            g = infer_columns(df); cols = list(df.columns); opt = ["(none)"] + cols
            ix = lambda c, o: o.index(c) if c in o else 0
            st.markdown("**Confirm column mapping** (suggestions are heuristic; please check)")
            m1, m2, m3, m4 = st.columns(4)
            sc = m1.selectbox("Sentence column", cols, index=ix(g["sentence"], cols))
            ic = m2.selectbox("ID column", opt, index=ix(g["id"], opt))
            lc = m3.selectbox("Ground-truth (biased / not biased) column", cols, index=ix(g["label"], cols))
            cc = m4.selectbox("Bias-category column", opt, index=ix(g["category"], opt))
            vals = sorted(df[lc].dropna().astype(str).str.strip().unique())
            ps, ns, us = suggest_label_sides(vals)
            st.markdown("**Map label values** (this only tells the app which value means biased; labels are not changed)")
            l1, l2 = st.columns(2)
            pos_v = l1.multiselect(f"Values meaning '{POS}'", vals, default=ps)
            neg_v = l2.multiselect(f"Values meaning '{NEG}'", vals, default=ns)
            if us: st.warning(f"Values not auto-assigned (select them above, or they will be excluded and reported): {us}")
            if set(pos_v) & set(neg_v):
                st.error("A value is mapped to both classes."); st.stop()
            mapping = {"sentence": sc, "id": None if ic == "(none)" else ic, "label": lc, "category": None if cc == "(none)" else cc}
            valid, excl, q = prepare_corpus(df, mapping, pos_v, neg_v)

            st.markdown("**Data-quality report (before any API call)**")
            a, b, c, d = st.columns(4)
            a.metric("Rows in sheet", q["n_rows_total"]); b.metric("Valid", q["n_valid"]); c.metric("Biased", q["n_biased"]); d.metric("Not biased", q["n_not_biased"])
            st.write(f"Excluded: **{q['n_excluded']}** (missing sentence {q['missing_sentence']}, missing label {q['missing_label']}, unmapped label {q['unmapped_label']}) · duplicate-sentence rows: **{q['duplicate_sentence_rows']}** · duplicate IDs: {q['duplicate_ids']} · conflicting duplicate groups: **{len(q['conflicting_duplicate_groups'])}**")
            if q["id_generated"]: st.info("No ID column selected: spreadsheet row numbers are used as IDs.")
            if q["label_variants"] or q["whitespace_padded_labels"]: st.warning(f"Label formatting inconsistencies: variants {q['label_variants']}, whitespace-padded labels {q['whitespace_padded_labels']}. Variants are mapped by your selection above; the raw label is kept in the output.")
            if q["category_spelling_variants"]: st.warning(f"Category spelling variants (treated as DIFFERENT categories until you fix your source): {q['category_spelling_variants']}")
            if q["biased_rows_without_category"]: st.warning(f"{q['biased_rows_without_category']} biased rows have no category and cannot be scored at category level.")
            if q["conflicting_duplicate_groups"]: st.json(q["conflicting_duplicate_groups"])
            st.write("Category distribution (biased rows):", q["category_counts"] or "none")
            if len(excl): st.dataframe(excl)

            cats_from_corpus = sorted(q["category_counts"])
            st.caption("Categories given to the detector (taxonomy only; per-sentence labels are never shown to it): " + ", ".join(cats_from_corpus))
            r1, r2 = st.columns(2)
            do_conv = r1.checkbox("Also generate inclusive alternatives for sentences the AI classifies as biased", value=True)
            limit = r2.number_input("Pilot run: process only the first N valid rows (0 = all)", 0, max(len(valid), 0), 0)
            if not cats_from_corpus: st.error("No categories found in the corpus; select the category column.")
            can_run = len(valid) > 0 and bool(cats_from_corpus) and (client.is_mock or client.key_present())
            if st.button("Run experiment", type="primary", disabled=not can_run):
                work = valid.head(int(limit)) if limit else valid
                qq = dict(q)
                if limit: st.warning(f"Pilot subset: {len(work)} of {len(valid)} valid rows. Do not report as the full experiment.")
                bar = st.progress(0.0, text="Processing…")
                res = run_experiment(work, client, cats_from_corpus, do_conv, workers, lambda i, n: bar.progress(i / n, text=f"Processing {i}/{n}"))
                info = {"filename": up.name, "sha256": runstore.sha256(raw), "sheet": sheet, "n_rows_in_sheet": q["n_rows_total"],
                        "mapping": mapping, "values_mapped_to_biased": pos_v, "values_mapped_to_not_biased": neg_v,
                        "pilot_subset_size": int(limit) or None}
                mf = build_manifest(client, cats_from_corpus, info, q, workers, do_conv)
                rd = runstore.new_run_dir()
                runstore.save_run(rd, res, mf, q, excl, raw, up.name)
                run = runstore.load_run(rd)
                A = compute_all(run["results"], run["quality"], None, None)
                write_workbook(rd / "results_workbook.xlsx", run, A); make_figures(A, run["quality"], rd)
                (rd / "results_discussion_draft.md").write_text(results_draft(run, A), encoding="utf-8")
                st.session_state["run"] = run
                st.success(f"Done. Saved to outputs/{rd.name}. Open the Results dashboard tab.")

# ---------------- tab 3 ----------------
def show_tbl(df, **k):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].astype(str)   # mixed number/"undefined" columns cannot be Arrow-serialised
    st.dataframe(d, **k)

with tab3:
    runs = runstore.list_runs()
    names = [p.name for p in runs]
    cur = st.session_state.get("run")
    pick = st.selectbox("Run", ["(current session run)" if cur else "(none)"] + names, index=0)
    if pick in names:
        st.session_state["run"] = cur = runstore.load_run(runstore.out_root() / pick)
    if not cur:
        st.info("No run yet. Upload and run a corpus in tab 2, or choose a saved run.")
    else:
        run = cur
        instrument = load_instrument(st.session_state.get("instrument"))
        ratings = runstore.load_ratings(run["dir"])
        A = compute_all(run["results"], run["quality"], ratings if len(ratings) else None, instrument)
        mf, q, res = run["manifest"], run["quality"], run["results"]
        if mf.get("is_mock"): st.error("This run used the MOCK provider. These numbers are software tests, not research results.")
        if mf["corpus"].get("pilot_subset_size"): st.warning(f"Pilot subset: {mf['corpus']['pilot_subset_size']} rows only.")
        st.markdown("### A. Dataset summary")
        a, b, c, d, e = st.columns(5)
        a.metric("Valid sentences", q["n_valid"]); b.metric("Biased", q["n_biased"]); c.metric("Not biased", q["n_not_biased"]); d.metric("Excluded/invalid", q["n_excluded"]); e.metric("Duplicate rows", q["duplicate_sentence_rows"])
        show_tbl(A["tables"]["Table 1. Corpus Distribution"])
        st.markdown("### B. Detection results")
        bm = A["binary"]
        cols = st.columns(8)
        for col, k in zip(cols, ["TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1"]): col.metric(k, str(fmt(bm[k], 3)))
        st.caption(f"Evaluated {bm['N_evaluated']} / {bm['N_total']}; API/parse failures: {bm['N_not_evaluated(API/parse errors)']}")
        for n in bm["Notes"]: st.warning(n)
        l, r = st.columns(2); l.markdown("**Confusion matrix**"); l.dataframe(A["confusion"]); r.markdown("**Classification report**"); r.dataframe(A["report"])
        st.markdown("### C. Category results"); show_tbl(A["category"])
        st.markdown("### D. Error analysis")
        st.write("Recurring patterns (descriptive counts):", A["patterns"]); show_tbl(A["errors"])
        st.markdown("### E. Conversion results")
        st.json(A["conversion_summary"])
        conv = res[res.Conversion_Status == "ok"][["ID", "Sentence", "Detected_Category", "Inclusive_Alternative", "Conversion_Notes"]]
        st.markdown("**Representative examples (first 10 in corpus order, not hand-picked)**"); show_tbl(conv.head(10))
        st.markdown("**Problematic outputs**"); show_tbl(A["conversion_problem"][["ID", "Sentence", "Inclusive_Alternative", "Conversion_Status", "Conversion_Notes"]])
        st.markdown("### F. Expert validation")
        if A["expert"] is None: st.info("No expert ratings yet (tab 4).")
        else:
            ex = A["expert"]; st.json(ex["overall"]); show_tbl(ex["criterion"]); show_tbl(ex["frequency"])
            st.markdown("**Agreement** (only items rated by 2+ experts)"); show_tbl(ex["agreement"]) if len(ex["agreement"]) else st.caption("Not computed: no item rated by 2+ raters.")
            st.markdown("**Comments**"); show_tbl(ex["comments"])
        st.markdown("### Research tables"); [ (st.markdown(f"**{k}**"), show_tbl(v)) for k, v in A["tables"].items() ]
        st.markdown("### Figures")
        figs = make_figures(A, q, run["dir"]); fc = st.columns(2)
        for i, (n, p) in enumerate(figs.items()): fc[i % 2].image(str(p), caption=n)
        st.markdown("### Reproducibility manifest"); st.json({k: v for k, v in mf.items() if k not in ("detection_system_prompt", "conversion_system_prompt")}); 
        with st.expander("Prompts used"):
            st.code(mf["detection_system_prompt"]); st.code(mf["detection_user_prompt_template"]); st.code(mf["conversion_system_prompt"]); st.code(mf["conversion_user_prompt_template"])
        st.markdown("### Downloads")
        if st.button("Rebuild workbook, figures and draft (include latest expert ratings)"):
            write_workbook(run["dir"] / "results_workbook.xlsx", run, A)
            (run["dir"] / "results_discussion_draft.md").write_text(results_draft(run, A), encoding="utf-8"); st.success("Rebuilt.")
        wb = run["dir"] / "results_workbook.xlsx"
        if not wb.exists(): write_workbook(wb, run, A)
        dl = st.columns(4)
        dl[0].download_button("Excel workbook", wb.read_bytes(), "results_workbook.xlsx")
        dl[1].download_button("Sentence-level CSV", res.to_csv(index=False).encode("utf-8-sig"), "results.csv")
        dl[2].download_button("Manifest JSON", json.dumps(mf, indent=2, default=str).encode(), "manifest.json")
        draft = results_draft(run, A); dl[3].download_button("Results & Discussion draft (.md)", draft.encode("utf-8"), "results_discussion_draft.md")
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w") as z:
            for p in figs.values(): z.write(p, p.name)
        st.download_button("Figures (zip)", zb.getvalue(), "figures.zip")

# ---------------- tab 4 ----------------
with tab4:
    run = st.session_state.get("run")
    if not run:
        st.info("Run an experiment (tab 2) or select a saved run in tab 3 first.")
    else:
        st.subheader("Expert validation of generated alternatives")
        iu = st.file_uploader("Optional: upload YOUR validation instrument (JSON, same structure as config/expert_instrument.json)", type=["json"])
        if iu is not None:
            st.session_state["instrument"] = json.loads(iu.getvalue().decode("utf-8"))
        ins = load_instrument(st.session_state.get("instrument"))
        if "PLACEHOLDER" in ins["name"]: st.warning("Using the default PLACEHOLDER instrument. Replace it with your own before collecting data for the paper.")
        st.caption(f"Instrument: {ins['name']} · scale {ins['scale_min']}–{ins['scale_max']}: " + ", ".join(f"{k}={v}" for k, v in ins["scale_labels"].items()))
        items = run["results"][run["results"].Conversion_Status == "ok"].reset_index(drop=True)
        if items.empty: st.info("No generated alternatives in this run.")
        else:
            rater = st.text_input("Expert ID / name (required)")
            idx = st.selectbox("Item", range(len(items)), format_func=lambda i: f"{i+1}/{len(items)} · ID {items.ID[i]}")
            it = items.iloc[idx]
            done = runstore.load_ratings(run["dir"])
            if rater and len(done[(done.rater_id == rater) & (done.item_key.astype(str) == str(it.Source_Row))]): st.info("You already rated this item; submitting again replaces your earlier ratings.")
            with st.container(border=True):
                st.markdown(f"**Original:** {it.Sentence}"); st.markdown(f"**Detected category:** {it.Detected_Category}  ·  **Why:** {it.Explanation}")
                st.markdown(f"**Suggested alternative:** {it.Inclusive_Alternative}")
            scale = list(range(int(ins["scale_min"]), int(ins["scale_max"]) + 1))
            with st.form(f"f_{idx}_{rater}"):
                got = {}
                for cr in ins["criteria"]:
                    got[cr["key"]] = st.radio(f"{cr['label']} — {cr.get('description','')}", scale, index=None, horizontal=True, format_func=lambda v: f"{v} ({ins['scale_labels'].get(str(v), '')})")
                comment = st.text_area("Comments (optional)")
                if st.form_submit_button("Submit ratings"):
                    if not rater.strip(): st.error("Enter your expert ID.")
                    elif any(v is None for v in got.values()): st.error("Rate every criterion.")
                    else:
                        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        runstore.append_ratings(run["dir"], [{"item_key": str(it.Source_Row), "item_id": it.ID, "rater_id": rater.strip(), "criterion": k, "rating": v, "comment": comment if i == 0 else "", "timestamp_utc": ts} for i, (k, v) in enumerate(got.items())])
                        st.success("Saved.")
            up2 = st.file_uploader("Or import ratings CSV (columns: item_key,item_id,rater_id,criterion,rating,comment,timestamp_utc)", type=["csv"])
            if up2 is not None and st.button("Import ratings"):
                imp = pd.read_csv(up2, keep_default_na=False)
                miss = set(runstore.RATING_COLS[:5]) - set(imp.columns)
                if miss: st.error(f"Missing columns: {miss}")
                else:
                    for c in runstore.RATING_COLS:
                        if c not in imp.columns: imp[c] = ""
                    runstore.append_ratings(run["dir"], imp[runstore.RATING_COLS].to_dict("records")); st.success("Imported.")
            cur_r = runstore.load_ratings(run["dir"])
            st.write(f"Ratings stored for this run: {len(cur_r)} ({cur_r.rater_id.nunique() if len(cur_r) else 0} rater(s)). Statistics appear in the dashboard (section F).")
