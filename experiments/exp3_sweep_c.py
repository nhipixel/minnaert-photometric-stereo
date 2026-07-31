"""
Stage 4. Angular error as a function of the Minnaert exponent.

A sphere is rendered across the full Minnaert range and the Lambertian solver
is run on all of it. The resulting curve has three points fixed in advance by
theory rather than by fitting:

    c = 1.0   Lambertian, exact recovery
    c = 0.5   simplified Hapke
    c = 0.0   the estimate collapses to a single direction everywhere

The first order law is plotted underneath the measurement. It is a prediction
made before the sweep runs, not a fit to it.

Measurements are taken on fully lit pixels only. The reduction holds where no
observation is clamped by attached shadow, so restricting to that set isolates
the reflectance effect from the shadow effect, which is studied separately.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brdf import render_minnaert
from geometry import sphere_normals
from lights import condition_number, cone_rig
from metrics import angular_error_deg, summarize
from results_io import figure_path, save_section
from solver import woodham_lstsq
from theory import collapse_error_deg, first_order_error_deg, fully_lit_mask

RESOLUTION = 256
N_LIGHTS = 10
SLANT_DEG = 35.0
C_VALUES = np.round(np.arange(0.0, 1.0001, 0.02), 4)

LAMBERTIAN_C = 1.0
HAPKE_C = 0.5


def sweep(resolution=RESOLUTION, m=N_LIGHTS, slant=SLANT_DEG, c_values=C_VALUES):
    """Measured and predicted error at each exponent. Returns parallel arrays."""
    normals, mask = sphere_normals(resolution)
    lights = cone_rig(m, slant)
    lit = fully_lit_mask(normals, lights, mask)

    measured_mean, measured_median, predicted = [], [], []
    for c in c_values:
        images = render_minnaert(normals, lights, float(c))
        est, _ = woodham_lstsq(images, lights, mask)

        stats = summarize(angular_error_deg(est, normals, lit), lit)
        measured_mean.append(stats["mean_deg"])
        measured_median.append(stats["median_deg"])
        predicted.append(float(np.nanmean(first_order_error_deg(normals, lights, float(c), lit))))

    return {
        "c": np.asarray(c_values, dtype=float),
        "mean": np.asarray(measured_mean),
        "median": np.asarray(measured_median),
        "predicted": np.asarray(predicted),
        "collapse": float(np.nanmean(collapse_error_deg(normals, lights, lit))),
        "n_pixels": int(lit.sum()),
        "cond_L": condition_number(lights),
    }


def _plot(data, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    ax.axhline(data["collapse"], color="0.6", ls=":", lw=1.2,
               label=f"collapse limit at c = 0 ({data['collapse']:.1f} deg)")
    ax.plot(data["c"], data["predicted"], color="tab:red", ls="--", lw=1.5,
            label="first order prediction")
    ax.plot(data["c"], data["mean"], color="tab:blue", lw=2.0, label="measured mean")
    ax.plot(data["c"], data["median"], color="tab:cyan", lw=1.4, label="measured median")

    # Labels sit low because the legend has to take the upper left.
    for c, name in [(LAMBERTIAN_C, "Lambertian"), (HAPKE_C, "simplified Hapke")]:
        ax.axvline(c, color="0.75", lw=0.9, zorder=0)
        ax.annotate(name, xy=(c, 0.0), xytext=(-4, 6),
                    textcoords="offset points", rotation=90,
                    ha="right", va="bottom", fontsize=8, color="0.35",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))

    ax.set_xlabel("Minnaert exponent c")
    ax.set_ylabel("angular error (degrees)")
    ax.set_xlim(1.0, 0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run():
    data = sweep()

    at = lambda target: float(data["mean"][int(np.argmin(np.abs(data["c"] - target)))])
    payload = {
        "resolution": RESOLUTION,
        "n_lights": N_LIGHTS,
        "slant_deg": SLANT_DEG,
        "cond_L": data["cond_L"],
        "n_pixels_fully_lit": data["n_pixels"],
        "mae_at_lambertian_deg": at(LAMBERTIAN_C),
        "mae_at_hapke_deg": at(HAPKE_C),
        "mae_at_c_zero_deg": at(0.0),
        "collapse_limit_deg": data["collapse"],
        # The median sits above the mean across the whole sweep. The error
        # distribution is left skewed because pixels whose normal aligns with
        # the light rig axis are biased far less than the rest, so a minority
        # of low error pixels pulls the mean down.
        "median_minus_mean_at_hapke_deg": float(
            data["median"][int(np.argmin(np.abs(data["c"] - HAPKE_C)))] - at(HAPKE_C)
        ),
        "curve": {
            "c": data["c"].tolist(),
            "mean_deg": data["mean"].tolist(),
            "median_deg": data["median"].tolist(),
            "predicted_deg": data["predicted"].tolist(),
        },
    }

    path = figure_path("fig2_minnaert_sweep.png")
    _plot(data, path)
    save_section("minnaert_sweep", payload)

    print(f"fully lit pixels : {data['n_pixels']}")
    print(f"cond(L)          : {data['cond_L']:.4f}")
    print(f"MAE at c = 1.0   : {payload['mae_at_lambertian_deg']:.3e} deg")
    print(f"MAE at c = 0.5   : {payload['mae_at_hapke_deg']:.3f} deg  (simplified Hapke)")
    print(f"MAE at c = 0.0   : {payload['mae_at_c_zero_deg']:.3f} deg")
    print(f"collapse limit   : {payload['collapse_limit_deg']:.3f} deg  (predicted)")
    print(f"figure           : {path}")
    return payload


if __name__ == "__main__":
    run()
