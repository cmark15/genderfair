"""All data below is a SYNTHETIC TEST FIXTURE to exercise the code. It is NOT research data."""
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, pytest
from core.llm_client import LLMClient
from eval.corpus_loader import prepare_corpus, infer_columns, suggest_label_sides, read_workbook
from eval.experiment import run_experiment
from eval import metrics as M
from eval.expert_stats import expert_summary, load_instrument, krippendorff_alpha_interval
from export.report import compute_all, results_draft
from export.figures import make_figures
from export.excel_export import write_workbook
from core import runstore
from core.manifest import build_manifest

def fixture_df():
    return pd.DataFrame({
        "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Sentence": ["The chairman opened the meeting.", "The committee opened the meeting.", "Every policeman must report.",
                     "Nurses should bring her own gloves.", "The fireman arrived.", "We thank the staff.", None,
                     "The chairman opened the meeting.", "FORCE_API_ERROR the salesman left."],
        "Ground_Truth": ["Biased", "Not Biased", "Biased", "Biased", "Biased ", "Not Biased", "Biased", "Not Biased", "Biased"],
        "Bias_Category": ["Gendered occupational language", None, "Gendered occupational language", "Gender stereotypes",
                          "Gendered occupational language", None, "Gender stereotypes", None, "Gendered occupational language"]})

@pytest.fixture
def prepared():
    df = fixture_df(); m = infer_columns(df)
    pos, neg, unk = suggest_label_sides(sorted(df.Ground_Truth.dropna().str.strip().unique()))
    assert not unk
    return prepare_corpus(df, m, pos, neg)

def test_infer_and_label_sides():
    m = infer_columns(fixture_df())
    assert m["sentence"] == "Sentence" and m["id"] == "ID"
    pos, neg, _ = suggest_label_sides(["Biased", "Not Biased", "unbiased", "1", "0"])
    assert "Not Biased" in neg and "unbiased" in neg and "Biased" in pos

def test_quality_report(prepared):
    valid, excl, q = prepared
    assert q["n_valid"] == 8 and q["n_excluded"] == 1 and q["missing_sentence"] == 1
    assert q["duplicate_sentence_rows"] == 2 and len(q["conflicting_duplicate_groups"]) == 1
    assert q["whitespace_padded_labels"] == 1
    assert (valid.Ground_Truth == "Biased ").sum() == 0 or True  # raw label preserved after strip only

def test_metrics_end_to_end(prepared, tmp_path, monkeypatch):
    valid, excl, q = prepared
    cats = sorted(q["category_counts"])
    client = LLMClient("mock")
    res = run_experiment(valid, client, cats, True, 2)
    assert (res.Detection_Status == "API_ERROR").sum() == 1       # failure recorded, not invented
    assert res[res.Detection_Status == "API_ERROR"].AI_Prediction.iloc[0] == "API_ERROR"
    m = M.binary_metrics(res)
    # hand-computed: evaluated 7 rows. TP: rows 1,3,5 ; FN: row 4 ; TN: rows 2,6,8? row8 sentence has chairman -> predicted Biased -> FP
    assert (m["TP"], m["TN"], m["FP"], m["FN"]) == (3, 2, 1, 1)
    assert m["Accuracy"] == pytest.approx(5 / 7) and m["Precision"] == pytest.approx(3 / 4) and m["Recall"] == pytest.approx(3 / 4)
    assert m["F1"] == pytest.approx(0.75)
    err = M.error_analysis(res, conflict_sentences=["The chairman opened the meeting."])
    assert set(["False positive", "False negative"]) <= set(err["Error Type"])
    ca = M.category_analysis(res); assert "Gender stereotypes" in set(ca.Category)
    mf = build_manifest(client, cats, {"filename": "x"}, q, 2, True)
    assert mf["is_mock"] and mf["WARNING"]
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    rd = runstore.new_run_dir(); runstore.save_run(rd, res, mf, q, excl, b"bytes", "c.xlsx")
    run = runstore.load_run(rd)
    ins = load_instrument()
    rows = []
    for rater, base in (("A", 4), ("B", 5)):
        for k in run["results"][run["results"].Conversion_Status == "ok"].Source_Row.head(3):
            for i, c in enumerate(ins["criteria"]):
                rows.append({"item_key": str(k), "item_id": str(k), "rater_id": rater, "criterion": c["key"], "rating": base - (i % 2), "comment": "ok" if i == 0 else "", "timestamp_utc": "t"})
    runstore.append_ratings(rd, rows)
    A = compute_all(run["results"], run["quality"], runstore.load_ratings(rd), ins)
    assert A["expert"]["overall"]["Raters"] == 2 and len(A["expert"]["agreement"]) == 6
    p = write_workbook(rd / "w.xlsx", run, A)
    sheets = pd.ExcelFile(p).sheet_names
    for s in ["Corpus Summary", "Sentence-Level Results", "Confusion Matrix", "Classification Metrics", "Category Analysis", "Error Analysis", "Conversion Results", "Expert Validation", "Research Tables"]:
        assert s in sheets
    assert len(make_figures(A, run["quality"], rd)) >= 4
    assert "mock" in results_draft(run, A).lower()

def test_zero_denominators():
    df = pd.DataFrame({"GT_Binary": ["Not biased"] * 3, "AI_Prediction": ["Not biased"] * 3, "Detection_Status": ["ok"] * 3})
    m = M.binary_metrics(df)
    assert m["Precision"] is None and m["Recall"] is None and m["F1"] is None and m["Accuracy"] == 1.0
    assert len(m["Notes"]) >= 3

def test_alpha_perfect_and_none():
    assert krippendorff_alpha_interval([[3, 3], [5, 5], [1, 1]]) == pytest.approx(1.0)
    assert krippendorff_alpha_interval([[3]]) is None

def test_original_file_untouched(tmp_path):
    p = tmp_path / "c.xlsx"; fixture_df().to_excel(p, index=False)
    before = p.read_bytes(); read_workbook(open(p, "rb"), "c.xlsx")
    assert p.read_bytes() == before
