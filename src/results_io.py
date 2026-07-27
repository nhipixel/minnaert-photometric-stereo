"""
Single source of truth for every number quoted in the report.

Experiments write here and the report reads only from here. Nothing is
transcribed by hand, which is what stops a table going stale after a rerun.
"""
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.json")


def _ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_all():
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def save_section(name, payload):
    """Replace one named section, leaving the other experiments untouched."""
    _ensure_dirs()
    data = load_all()
    data[name] = payload
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return RESULTS_PATH


def figure_path(filename):
    _ensure_dirs()
    return os.path.join(FIGURES_DIR, filename)
