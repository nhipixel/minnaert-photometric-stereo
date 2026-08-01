"""
Reproduced evaluation of the published normal maps shipped with the benchmark.

These are not this project's results. Eight published calibrated methods plus
the least squares baseline are rescored under one metric, with degenerate
ground truth pixels excluded, so every row is comparable to every other row
and to the rest of the report. The table this produces is the yardstick any
correction has to be read against.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from diligent import (
    OBJECTS,
    PUBLISHED_METHODS,
    is_available,
    load_ground_truth,
    load_published_estimate,
)
from metrics import mean_angular_error
from results_io import save_section


def run():
    if not is_available():
        print("benchmark not found, skipping. See README for the download.")
        return None

    truth = {name: load_ground_truth(name) for name in OBJECTS}

    table = {}
    for method in PUBLISHED_METHODS:
        row = {}
        for name in OBJECTS:
            gt, mask = truth[name]
            est = load_published_estimate(name, method)
            row[name] = mean_angular_error(est, gt, mask)
        row["average"] = float(np.mean([row[n] for n in OBJECTS]))
        table[method] = row

    ranked = sorted(PUBLISHED_METHODS, key=lambda m: table[m]["average"])
    hdr = f"{'method':<16}" + "".join(f"{o[:7]:>8}" for o in OBJECTS) + f"{'avg':>8}"
    print(hdr)
    for method in ranked:
        row = table[method]
        print(f"{method:<16}" + "".join(f"{row[o]:>8.2f}" for o in OBJECTS)
              + f"{row['average']:>8.2f}")

    payload = {
        "per_method": table,
        "ranking_by_average": ranked,
        "best_method": ranked[0],
        "best_average_deg": table[ranked[0]]["average"],
        "baseline_average_deg": table["l2"]["average"],
        "note": (
            "Reproduced evaluation of estimate maps distributed with the "
            "benchmark, scored with this project's metric. Degenerate ground "
            "truth pixels are excluded, so the pot2 column sits slightly below "
            "originally published values for every method."
        ),
    }
    save_section("landscape", payload)
    return payload


if __name__ == "__main__":
    run()
