
import os
import json
import hashlib
import datetime
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parent.parent


def out_root():
    p = Path(os.getenv("OUTPUT_DIR", "outputs"))
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_run_dir():
    rid = datetime.datetime.now(datetime.timezone.utc).strftime(
        "run_%Y%m%d_%H%M%S"
    )
    d = out_root() / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(b: bytes):
    return hashlib.sha256(b).hexdigest()


def read_csv_or_empty(path, columns=None, **kwargs):
    """Read a CSV safely, including when the file is empty."""
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)


def save_run(
    run_dir,
    results,
    manifest,
    quality,
    excluded,
    corpus_bytes=None,
    corpus_name=None,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        run_dir / "results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    excluded.to_csv(
        run_dir / "excluded_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    (run_dir / "quality.json").write_text(
        json.dumps(quality, indent=2, default=str),
        encoding="utf-8",
    )

    if corpus_bytes is not None:
        suffix = Path(corpus_name or "corpus.xlsx").suffix
        (run_dir / f"input_corpus_copy{suffix}").write_bytes(
            corpus_bytes
        )


def load_run(run_dir):
    d = Path(run_dir)

    # Results are required for a valid saved run.
    res = pd.read_csv(
        d / "results.csv",
        dtype={"ID": str},
        keep_default_na=False,
    )

    res["Confidence_SelfReported"] = pd.to_numeric(
        res["Confidence_SelfReported"],
        errors="coerce",
    )

    for c in ("Source_Row",):
        res[c] = pd.to_numeric(res[c], errors="coerce")

    # Excluded records are optional and may be empty.
    excl = read_csv_or_empty(
        d / "excluded_records.csv",
        keep_default_na=False,
    )

    manifest = json.loads(
        (d / "manifest.json").read_text(encoding="utf-8")
    )

    quality = json.loads(
        (d / "quality.json").read_text(encoding="utf-8")
    )

    return {
        "dir": d,
        "results": res,
        "manifest": manifest,
        "quality": quality,
        "excluded": excl,
    }


def list_runs():
    return sorted(
        [
            p
            for p in out_root().iterdir()
            if (p / "results.csv").exists()
        ],
        reverse=True,
    )


def ratings_path(run_dir):
    return Path(run_dir) / "expert_ratings.csv"


RATING_COLS = [
    "item_key",
    "item_id",
    "rater_id",
    "criterion",
    "rating",
    "comment",
    "timestamp_utc",
]


def load_ratings(run_dir):
    return read_csv_or_empty(
        ratings_path(run_dir),
        columns=RATING_COLS,
        keep_default_na=False,
    )


def append_ratings(run_dir, rows):
    p = ratings_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)

    new = pd.DataFrame(rows, columns=RATING_COLS)

    cur = read_csv_or_empty(
        p,
        columns=RATING_COLS,
        keep_default_na=False,
    )

    # A rater re-rating the same item/criterion
    # replaces the earlier entry.
    allr = pd.concat([cur, new], ignore_index=True)

    allr = allr.drop_duplicates(
        subset=["item_key", "rater_id", "criterion"],
        keep="last",
    )

    allr.to_csv(
        p,
        index=False,
        encoding="utf-8-sig",
    )
