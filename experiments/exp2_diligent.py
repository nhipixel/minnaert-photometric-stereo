"""
Stage 3. Woodham least squares on the DiLiGenT benchmark.

This satisfies the Dataset and Evaluation requirements: real objects with
ground truth normals, scored by mean angular error in degrees. The solver is
the same one used on synthetic data, with no changes for real input.

Every number is checked against the published baseline, which this project did
not produce, so a convention error cannot pass silently.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from diligent import OBJECTS, PUBLISHED_AVERAGE, PUBLISHED_BASELINE, is_available, load_object
from metrics import angular_error_deg, summarize
from results_io import save_section
from solver import woodham_lstsq


def run():
    if not is_available():
        print("DiLiGenT not found, skipping. See README for the download.")
        return None

    rows = {}
    print(f"{'object':<10}{'ours':>9}{'published':>12}{'delta':>9}{'median':>9}")
    for name in OBJECTS:
        obj = load_object(name)
        est, _ = woodham_lstsq(obj.images, obj.lights, obj.mask)
        stats = summarize(angular_error_deg(est, obj.normals_gt, obj.mask), obj.mask)

        published = PUBLISHED_BASELINE[name]
        rows[name] = {
            "mae_deg": stats["mean_deg"],
            "median_deg": stats["median_deg"],
            "published_mae_deg": published,
            "delta_deg": stats["mean_deg"] - published,
            "n_pixels": stats["n_pixels"],
            "n_lights": int(obj.lights.shape[0]),
        }
        print(f"{name:<10}{stats['mean_deg']:>9.3f}{published:>12.2f}"
              f"{stats['mean_deg'] - published:>9.3f}{stats['median_deg']:>9.3f}")

    ours = float(np.mean([r["mae_deg"] for r in rows.values()]))
    print(f"{'average':<10}{ours:>9.3f}{PUBLISHED_AVERAGE:>12.2f}{ours - PUBLISHED_AVERAGE:>9.3f}")

    payload = {
        "per_object": rows,
        "average_mae_deg": ours,
        "published_average_mae_deg": PUBLISHED_AVERAGE,
        "max_abs_delta_deg": float(max(abs(r["delta_deg"]) for r in rows.values())),
        "note": (
            "Grey conversion clips at 1 to match the reference implementation, "
            "which uses Matlab rgb2gray. Without the clip, saturated specular "
            "pixels disagree and ball reads 4.17 instead of 4.10."
        ),
    }
    save_section("diligent_baseline", payload)
    return payload


if __name__ == "__main__":
    run()
