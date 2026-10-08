# Gender-Bias Detection & Inclusive-Language Conversion (research prototype)

## Run it
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY with LLM_PROVIDER=openai   (Windows: set / $env:)
streamlit run app.py
```
Keys come from environment variables (or a local `.env`, see `.env.example`) and are never shown, stored or logged.
Optionally set `LLM_MODEL` and `LLM_TEMPERATURE` (default 0). Model and temperature are recorded in every run's manifest.

## Upload your corpus
1. Open tab **2 · Corpus experiment** and click **Browse files** (accepts .xlsx, .xls, .csv).
2. Pick the sheet. Check the suggested columns (sentence, ID, ground-truth, category) and which label values mean *Biased* / *Not biased*.
3. Read the **data-quality report** (missing, duplicates, conflicting labels, spelling variants). Nothing is deleted or relabelled; unusable rows are listed as excluded.
4. Tip: run a **pilot** (first N rows) to check cost and output, then run with 0 = all rows.
5. Click **Run experiment**. Results appear in tab **3 · Results dashboard**.

Your file is only read in memory. A byte-for-byte copy and its SHA-256 are stored with the run.

## What each run saves (`outputs/run_<UTC timestamp>/`)
`results.csv` (sentence level) · `results_workbook.xlsx` (9 sheets) · `figures/*.png` · `results_discussion_draft.md` ·
`manifest.json` (model, parameters, prompts, corpus hash, library versions) · `quality.json` · `excluded_records.csv` · `input_corpus_copy.*` · `expert_ratings.csv`

## Reproduce an experiment
Use the same corpus (check the SHA-256 in `manifest.json`), same provider/model/temperature, and the prompts in `core/prompts.py` (full text is in the manifest). LLM output can still vary slightly between calls even at temperature 0, so report that caveat and consider repeated runs.

## Expert validation
Tab 4 uses `config/expert_instrument.json`, a **placeholder**. Replace it with your instrument (upload a JSON of the same shape in tab 4). Ratings are saved per run; statistics (frequency, %, mean, weighted mean, SD, and agreement when 2+ raters rate the same item) appear in dashboard section F. Press **Rebuild workbook** after collecting ratings.

## Notes on method
* The detector is told the category names found in your corpus (taxonomy only), never any sentence's label. Add your annotation guidelines to `config/annotation_guidelines.txt` to include them in the prompt (they are recorded in the manifest).
* API/parse failures are recorded as such and excluded from metrics.
* Confidence is self-reported by the LLM, not a calibrated probability.
* "Conversion succeeded" means a valid rewrite was returned, not that it is good; quality comes from expert ratings.
* `provider = mock` is a rule-based test double for software testing only; runs made with it are watermarked.

## Tests
`python -m pytest -q tests` (uses a synthetic fixture, not research data).
