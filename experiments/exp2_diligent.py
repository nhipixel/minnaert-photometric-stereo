"""
Woodham least squares on the DiLiGenT benchmark.

Real objects with ground truth normals, scored by mean angular error in
degrees. The solver is the same one used on synthetic data, with no changes for
real input.

Every number is checked against the published baseline, which this project did
not produce, so a convention error cannot pass silently.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from diligent import (
    BIT_DEPTH_SCALE,
    LUMA_BT601,
    OBJECTS,
    PUBLISHED_AVERAGE,
    PUBLISHED_BASELINE,
    is_available,
    load_object,
    load_official_l2,
    default_root,
    _read_matrix,
    _read_png,
)
from metrics import angular_error_deg, summarize
from results_io import save_section
from solver import woodham_lstsq


def clip_impact(name="ball"):
    """
    What the unit clamp in the grey conversion actually costs.

    The report quotes the size of this effect, so it is measured rather than
    asserted: the fraction of observations the clamp touches, and how far the
    estimate moves without it.
    """
    obj = load_object(name)
    folder = os.path.join(default_root(), f"{name}PNG")

    with open(os.path.join(folder, "filenames.txt"), encoding="utf-8") as fh:
        names = [line.strip() for line in fh if line.strip()]
    intensities = _read_matrix(os.path.join(folder, "light_intensities.txt"), 3)

    unclipped = np.empty_like(obj.images)
    n_clipped = 0
    for j, (fname, intensity) in enumerate(zip(names, intensities)):
        img = _read_png(os.path.join(folder, fname)) / BIT_DEPTH_SCALE
        grey = np.maximum(img / intensity, 0.0) @ LUMA_BT601
        unclipped[..., j] = grey
        n_clipped += int((grey[obj.mask] > 1.0).sum())

    est_clipped, _ = woodham_lstsq(obj.images, obj.lights, obj.mask)
    est_raw, _ = woodham_lstsq(unclipped, obj.lights, obj.mask)

    stats = summarize(angular_error_deg(est_clipped, est_raw, obj.mask), obj.mask)
    total_obs = int(obj.mask.sum()) * obj.images.shape[2]
    return {
        "object": name,
        "clipped_fraction_percent": 100.0 * n_clipped / total_obs,
        "max_disagreement_deg": stats["max_deg"],
        "mae_with_clip_deg": summarize(
            angular_error_deg(est_clipped, obj.normals_gt, obj.mask), obj.mask)["mean_deg"],
        "mae_without_clip_deg": summarize(
            angular_error_deg(est_raw, obj.normals_gt, obj.mask), obj.mask)["mean_deg"],
    }


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

        # Agreement with the shipped reference estimate, which is a stronger
        # check than matching its mean, and the count of pixels whose ground
        # truth carries no direction and so cannot be scored.
        official = summarize(
            angular_error_deg(est, load_official_l2(name), obj.mask), obj.mask
        )["mean_deg"]
        n_scored = stats["n_pixels"]
        n_masked = int(obj.mask.sum())

        published = PUBLISHED_BASELINE[name]
        rows[name] = {
            "mae_deg": stats["mean_deg"],
            "median_deg": stats["median_deg"],
            "published_mae_deg": published,
            "delta_deg": stats["mean_deg"] - published,
            "n_pixels": n_scored,
            "n_degenerate_gt_pixels": n_masked - n_scored,
            "agreement_with_official_deg": official,
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
        "max_agreement_with_official_deg": float(
            max(r["agreement_with_official_deg"] for r in rows.values())
        ),
        "total_degenerate_gt_pixels": int(
            sum(r["n_degenerate_gt_pixels"] for r in rows.values())
        ),
        "clip_impact": clip_impact(),
        "note": (
            "Grey conversion clips at 1 to match the reference implementation. "
            "Without the clip, saturated specular pixels disagree and ball "
            "reads 4.17 instead of 4.10. Degenerate ground truth pixels are "
            "excluded from the mean, since their true orientation is undefined."
        ),
    }
    save_section("diligent_baseline", payload)
    return payload


if __name__ == "__main__":
    run()
