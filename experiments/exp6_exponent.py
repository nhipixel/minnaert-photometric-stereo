"""
Effective exponent of the real objects, and the correction study.

Four questions, in order:

  1. How Minnaert is each object? The per pixel exponent fit against the
     ground truth normals gives a per object median c_eff. This is a
     diagnostic, since it reads the answer key; it measures the material, it
     is not a method.
  2. Does the synthetic curve transfer? For each object the sphere is rendered
     at that object's own c_eff under that object's own 96 light rig, and the
     resulting error is compared with the measured one. Using the object's own
     lights matters because the error magnitude is rig dependent.
  3. What would the correction buy if the exponent were known? Raising each
     observation to 1/c and rerunning the unchanged solver is exact on
     synthetic data by construction. On real data with the oracle per pixel
     exponent it bounds what any estimator of c could achieve.
  4. Can the exponent be estimated without ground truth? The obvious
     alternation, fit against the current estimate then resolve, is run and
     reported. Trimmed least squares is run alongside as the robust control
     that uses no reflectance model at all.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brdf import render_minnaert
from diligent import OBJECTS, is_available, load_object
from exponent import apply_exponent_correction, fit_exponent_map
from geometry import sphere_normals
from metrics import angular_error_deg, mean_angular_error
from results_io import figure_path, save_section
from solver import woodham_lstsq, woodham_trimmed
from theory import fully_lit_mask

SPHERE_RES = 192
NAIVE_ROUNDS = 3

DIFFUSE = ("ball", "cat", "pot1", "bear")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5,
})


def rig_matched_prediction(lights, c):
    """Sphere error at exponent c under the object's own light rig."""
    normals, mask = sphere_normals(SPHERE_RES)
    lit = fully_lit_mask(normals, lights, mask)
    images = render_minnaert(normals, lights, float(c))
    est, _ = woodham_lstsq(images, lights, mask)
    return mean_angular_error(est, normals, lit)


def study_object(obj):
    images, lights, mask, gt = obj.images, obj.lights, obj.mask, obj.normals_gt

    base_est, _ = woodham_lstsq(images, lights, mask)
    base = mean_angular_error(base_est, gt, mask)

    c_map = fit_exponent_map(images, gt, lights)
    c_eff = float(np.median(c_map[mask]))

    oracle_pix_est, _ = woodham_lstsq(
        apply_exponent_correction(images, np.where(mask, c_map, 1.0)), lights, mask
    )
    oracle_obj_est, _ = woodham_lstsq(
        apply_exponent_correction(images, c_eff), lights, mask
    )

    cur = base_est
    for _ in range(NAIVE_ROUNDS):
        c_iter = fit_exponent_map(images, cur, lights)
        cur, _ = woodham_lstsq(apply_exponent_correction(images, c_iter), lights, mask)

    trim_est, _ = woodham_trimmed(images, lights, mask)

    return {
        "c_eff_median": c_eff,
        "c_eff_iqr": [float(np.percentile(c_map[mask], 25)),
                      float(np.percentile(c_map[mask], 75))],
        "mae_base_deg": base,
        "mae_rig_matched_synthetic_deg": rig_matched_prediction(lights, c_eff),
        "mae_oracle_pixel_deg": mean_angular_error(oracle_pix_est, gt, mask),
        "mae_oracle_object_deg": mean_angular_error(oracle_obj_est, gt, mask),
        "mae_naive_iterated_deg": mean_angular_error(cur, gt, mask),
        "mae_trimmed_deg": mean_angular_error(trim_est, gt, mask),
    }


SHOWCASE = ("ball", "cat", "harvest")


def _bbox(mask, margin=6):
    """Tight crop around the object. Uncropped frames waste most of the page."""
    rows, cols = np.where(mask)
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin + 1, mask.shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin + 1, mask.shape[1])
    return slice(r0, r1), slice(c0, c1)


def _plot_qualitative(path, names=SHOWCASE):
    """
    Real objects, shown rather than only tabulated.

    The exponent map is the point of this figure. Section 4.6 argues the
    departure from Lambertian varies across a surface, and a per object median
    cannot show that; the map can.

    The displayed input is the brightest of the 96 rather than the first, which
    is often a grazing light, and it is percentile stretched. Both are display
    choices only; nothing here feeds a number.
    """
    fig, axes = plt.subplots(len(names), 4, figsize=(6.6, 1.55 * len(names)))
    err_im = c_im = None

    for row, name in enumerate(names):
        obj = load_object(name)
        est, _ = woodham_lstsq(obj.images, obj.lights, obj.mask)
        err = angular_error_deg(est, obj.normals_gt, obj.mask)
        c_map = np.where(obj.mask, fit_exponent_map(obj.images, obj.normals_gt, obj.lights), np.nan)

        rs, cs = _bbox(obj.mask)
        mask_c = obj.mask[rs, cs]

        brightest = int(np.argmax(obj.images[obj.mask].sum(axis=0)))
        shown = obj.images[rs, cs, brightest]
        hi = np.percentile(shown[mask_c], 99.5)
        axes[row, 0].imshow(np.clip(shown / max(hi, 1e-12), 0, 1), cmap="gray")

        rgb = np.ones(est[rs, cs].shape)
        rgb[mask_c] = (est[rs, cs][mask_c] + 1.0) / 2.0
        axes[row, 1].imshow(np.clip(rgb, 0, 1))

        err_im = axes[row, 2].imshow(np.where(mask_c, err[rs, cs], np.nan),
                                     cmap="viridis", vmin=0, vmax=40)
        c_im = axes[row, 3].imshow(c_map[rs, cs], cmap="coolwarm", vmin=0.8, vmax=2.0)
        axes[row, 0].set_ylabel(name, fontsize=8)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for col, title in enumerate(["brightest of 96 inputs", "recovered $\\hat{n}$",
                                 "angular error (deg)", "$c_{\\mathrm{eff}}$"]):
        axes[0, col].set_title(title, fontsize=8)

    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(right=0.88)
    cax_e = fig.add_axes([0.90, 0.55, 0.012, 0.33])
    cax_c = fig.add_axes([0.90, 0.12, 0.012, 0.33])
    fig.colorbar(err_im, cax=cax_e).ax.tick_params(labelsize=6)
    fig.colorbar(c_im, cax=cax_c).ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot(rows, path):
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    # Labels are nudged individually because several objects cluster tightly
    # near the origin and default placement overlaps them illegibly.
    offsets = {"bear": (4, -7), "pot1": (4, 3), "cat": (4, -8), "ball": (5, -2),
               "buddha": (-4, 5), "pot2": (4, -7), "reading": (4, 3),
               "goblet": (4, -7), "cow": (5, 0), "harvest": (5, 0)}
    for name, r in rows.items():
        diffuse = name in DIFFUSE
        ax.scatter(r["mae_rig_matched_synthetic_deg"], r["mae_base_deg"], s=16,
                   color="tab:blue" if diffuse else "tab:red",
                   marker="o" if diffuse else "s", zorder=3)
        ax.annotate(name, (r["mae_rig_matched_synthetic_deg"], r["mae_base_deg"]),
                    xytext=offsets.get(name, (4, 3)), textcoords="offset points",
                    fontsize=6)
    lim = max(max(r["mae_base_deg"] for r in rows.values()),
              max(r["mae_rig_matched_synthetic_deg"] for r in rows.values())) * 1.1
    ax.plot([0, lim], [0, lim], color="0.7", lw=0.8, zorder=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("sphere error at $c_{\\mathrm{eff}}$, same rig (deg)")
    ax.set_ylabel("measured error (deg)")
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def run():
    if not is_available():
        print("benchmark not found, skipping. See README for the download.")
        return None

    rows = {}
    print(f"{'object':<9}{'c_eff':>7}{'base':>8}{'rigpred':>9}{'oracPix':>9}"
          f"{'oracObj':>9}{'naive':>8}{'trim':>8}")
    for name in OBJECTS:
        rows[name] = study_object(load_object(name))
        r = rows[name]
        print(f"{name:<9}{r['c_eff_median']:>7.3f}{r['mae_base_deg']:>8.2f}"
              f"{r['mae_rig_matched_synthetic_deg']:>9.2f}{r['mae_oracle_pixel_deg']:>9.2f}"
              f"{r['mae_oracle_object_deg']:>9.2f}{r['mae_naive_iterated_deg']:>8.2f}"
              f"{r['mae_trimmed_deg']:>8.2f}")

    avg = lambda key: float(np.mean([r[key] for r in rows.values()]))
    payload = {
        "per_object": rows,
        "avg_base_deg": avg("mae_base_deg"),
        "avg_oracle_pixel_deg": avg("mae_oracle_pixel_deg"),
        "avg_oracle_object_deg": avg("mae_oracle_object_deg"),
        "avg_naive_iterated_deg": avg("mae_naive_iterated_deg"),
        "avg_trimmed_deg": avg("mae_trimmed_deg"),
        "naive_rounds": NAIVE_ROUNDS,
        "sphere_resolution": SPHERE_RES,
        "note": (
            "c_eff is fit against ground truth normals and is a material "
            "diagnostic, not a method. The oracle corrections read the same "
            "fit and bound what an estimator of c could deliver. The naive "
            "alternation and the trimmed control are the legal comparisons."
        ),
    }
    path = figure_path("fig5_bridge.png")
    _plot(rows, path)
    _plot_qualitative(figure_path("fig6_real_objects.png"))
    save_section("exponent", payload)

    print(f"{'average':<9}{'':>7}{payload['avg_base_deg']:>8.2f}{'':>9}"
          f"{payload['avg_oracle_pixel_deg']:>9.2f}{payload['avg_oracle_object_deg']:>9.2f}"
          f"{payload['avg_naive_iterated_deg']:>8.2f}{payload['avg_trimmed_deg']:>8.2f}")
    print(f"figure : {path}")
    return payload


if __name__ == "__main__":
    run()
