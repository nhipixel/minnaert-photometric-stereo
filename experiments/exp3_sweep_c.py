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


# A two column page gives each figure about 3.25 inches. Authoring at final
# print size keeps the type at its intended point size; authoring larger and
# letting LaTeX shrink it renders the labels at roughly 5pt.
COLUMN_WIDTH_IN = 3.25
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.3,
})


def _plot(data, path):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_WIDTH_IN * 0.72))

    ax.axhline(data["collapse"], color="0.6", ls=":", lw=1.0,
               label=f"collapse limit ({data['collapse']:.1f}$^\\circ$)")
    ax.plot(data["c"], data["predicted"], color="tab:red", ls="--", lw=1.2,
            label="first order prediction")
    ax.plot(data["c"], data["mean"], color="tab:blue", lw=1.7, label="measured mean")
    ax.plot(data["c"], data["median"], color="tab:cyan", lw=1.1, label="measured median")

    # Labels sit low because the legend has to take the upper left.
    ax.set_xlabel("Minnaert exponent $c$")
    ax.set_ylabel("angular error (degrees)")
    ax.set_xlim(1.0, 0.0)
    ax.set_ylim(bottom=0.0)

    # Only the simplified Hapke point is marked. The Lambertian point sits on
    # the axis boundary, where any label collides with the tick labels, and the
    # curve starting at zero already makes it evident.
    top = ax.get_ylim()[1]
    ax.axvline(HAPKE_C, color="0.75", lw=0.8, zorder=0)
    ax.annotate("simpl. Hapke", xy=(HAPKE_C, top), xytext=(-3, -3),
                textcoords="offset points", rotation=90,
                ha="right", va="top", fontsize=6, color="0.35",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.85))

    ax.legend(loc="lower right", framealpha=0.9, borderpad=0.4, handlelength=1.6)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def make_teaser(path, resolution=256, m=N_LIGHTS, slant=SLANT_DEG):
    """
    One rendered view, the recovered normal map, and the error map, at five
    exponents. The visual companion to the sweep curve.
    """
    normals, mask = sphere_normals(resolution)
    lights = cone_rig(m, slant)
    cs = [1.0, 0.75, 0.5, 0.25, 0.0]

    fig, axes = plt.subplots(3, len(cs), figsize=(6.6, 3.9))
    last = None
    for col, c in enumerate(cs):
        images = render_minnaert(normals, lights, float(c))
        est, _ = woodham_lstsq(images, lights, mask)
        err = angular_error_deg(est, normals, mask)

        shown = images[..., 0]
        shown = shown / max(shown.max(), 1e-12)
        axes[0, col].imshow(np.where(mask, shown, np.nan), cmap="gray", vmin=0, vmax=1)

        rgb = np.ones(normals.shape)
        rgb[mask] = (est[mask] + 1.0) / 2.0
        axes[1, col].imshow(np.clip(rgb, 0, 1))

        last = axes[2, col].imshow(np.where(mask, err, np.nan),
                                   cmap="viridis", vmin=0, vmax=40)
        axes[0, col].set_title(f"$c = {c:.2f}$", fontsize=8)

    for row, label in enumerate(["rendering", "recovered $\\hat{n}$", "error (deg)"]):
        axes[row, 0].set_ylabel(label, fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(last, ax=axes[2, :], fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def albedo_bias(path, resolution=256, m=N_LIGHTS, slant=SLANT_DEG):
    """
    The other half of the invariance claim. The per pixel factor that cancels
    in the direction survives in the albedo, so the recovered albedo follows
    n_z^(c-1) while the normals stay exact. Slopes are computed on the isolated
    factor, where the relation is exact, and the figure shows the raw cloud.
    """
    normals, mask = sphere_normals(resolution)
    lights = cone_rig(m, slant)
    lit = fully_lit_mask(normals, lights, mask)
    nz = normals[..., 2][lit]

    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    slopes = {}
    for c, color in [(1.0, "tab:blue"), (0.5, "tab:red")]:
        images = render_minnaert(normals, lights, c)
        _, rho = woodham_lstsq(images, lights, mask)

        w = np.clip(normals @ lights.T, 0.0, None) ** c
        h = np.linalg.norm(w @ np.linalg.pinv(lights).T, axis=-1)
        isolated = rho[lit] / h[lit]
        slope = float(np.polyfit(np.log(nz), np.log(isolated), 1)[0])
        slopes[f"c{c}"] = slope

        sub = slice(None, None, 37)
        ax.plot(nz[sub], rho[lit][sub], ".", ms=1.5, color=color, alpha=0.5,
                label=f"$c = {c:.1f}$ (slope {slope:+.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n_z$")
    ax.set_ylabel("recovered albedo")
    ax.legend(framealpha=0.9, markerscale=6)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return slopes


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

    # Prediction quality at the two marked points, quoted in the report so the
    # validity range of the first order law is stated with numbers.
    for target, key in [(HAPKE_C, "hapke"), (0.0, "zero")]:
        i = int(np.argmin(np.abs(data["c"] - target)))
        payload[f"measured_over_predicted_at_{key}"] = float(
            data["mean"][i] / data["predicted"][i]
        )

    path = figure_path("fig2_minnaert_sweep.png")
    _plot(data, path)
    make_teaser(figure_path("fig1_teaser.png"))
    payload["albedo_slopes"] = albedo_bias(figure_path("fig3_albedo_bias.png"))
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
