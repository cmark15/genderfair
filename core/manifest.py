import sys, platform, datetime, importlib
from .prompts import (DETECTION_PROMPT_VERSION, CONVERSION_PROMPT_VERSION, detection_system, detection_user,
                      CONVERSION_SYSTEM, conversion_user, load_guidelines)

def _v(pkg):
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return "not installed"

def build_manifest(client, categories, corpus_info, quality, workers, convert, evaluation_note=None):
    return {
        "experiment_started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "WARNING": "MOCK PROVIDER - rule-based test double. Results are NOT valid research findings." if client.is_mock else None,
        "is_mock": client.is_mock,
        "provider": client.provider, "model_requested": client.model, "model_returned_by_api": client.model_returned,
        "api_version_note": "Anthropic SDK default 'anthropic-version' header applies (see SDK version)." if client.provider == "anthropic" else "OpenAI API (versioned by SDK).",
        "temperature_requested": client.temperature, "temperature_applied": client.temperature_applied,
        "max_tokens": client.max_tokens, "retries": client.retries, "parallel_workers": workers,
        "prompt_versions": {"detection": DETECTION_PROMPT_VERSION, "conversion": CONVERSION_PROMPT_VERSION},
        "detection_system_prompt": detection_system(categories), "detection_user_prompt_template": detection_user("{sentence}"),
        "conversion_system_prompt": CONVERSION_SYSTEM, "conversion_user_prompt_template": conversion_user("{sentence}", "{category}", "{explanation}", "{flagged_phrase}"),
        "annotation_guidelines_in_prompt": load_guidelines() or None,
        "categories_given_to_detector": categories, "conversion_enabled": convert,
        "corpus": corpus_info, "corpus_records_valid": quality["n_valid"], "corpus_records_excluded": quality["n_excluded"],
        "evaluation_methodology": evaluation_note or (
            "Binary detection scored against human ground truth (positive class = Biased); records with API/parse failures are "
            "excluded from metrics and reported. Category-level scoring uses exact match (case/whitespace-insensitive) between the "
            "detector's category and the ground-truth category on sentences both label as biased. Ground-truth labels are never altered."),
        "software": {"python": sys.version.split()[0], "platform": platform.platform(), "pandas": _v("pandas"),
                     "openpyxl": _v("openpyxl"), "matplotlib": _v("matplotlib"), "streamlit": _v("streamlit"),
                     "anthropic": _v("anthropic"), "openai": _v("openai")},
    }
