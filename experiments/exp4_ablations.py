"""
Stage 5. What else moves the error, besides the reflectance exponent.

Five questions, each with an expectation stated before the run:

  1. Conditioning. Squeezing the light azimuths into a narrow wedge should
     raise cond(L) and amplify noise monotonically with it. The growth is not
     expected to be proportional, and the most degenerate rigs saturate rather
     than continuing to scale, so the fitted slope is reported alongside.
  2. Light count. With fixed noise, error should fall roughly as 1/sqrt(m).
  3. Noise level. In the small noise regime error should grow linearly. Each
     level draws an independent realization, because rescaling one draw would
     make the fitted exponent exactly one by construction.
  4. Shadow handling. Dropping observations below the local horizon should
     help, since a clamped observation does not satisfy the linear model.
     Whether it still helps once the surface is non Lambertian is swept, not
     assumed.
  5. Rig dependence. The headline error at the simplified Hapke point is not a
     universal constant, so its spread across rigs is measured and reported.

Conditioning and light count are only visible under noise. With exact data the
solve is exact at any conditioning, so every experiment here that varies the
rig also injects noise.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brdf import add_noise, render_lambertian, render_minnaert
from geometry import bump_normals, sphere_normals
from lights import condition_number, cone_rig, near_coplanar_rig
from metrics import mean_angular_error
from results_io import figure_path, save_section
from solver import woodham_lstsq, woodham_shadow_aware
from theory import fully_lit_mask

RES = 128
SEED = 12345
NOISE = 0.01
PAGE_WIDTH_IN = 6.6

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "lines.linewidth": 1.3,
})


def _mae(images, lights, normals, mask, solver=woodham_lstsq):
    est, _ = solver(images, lights, mask)
    return mean_angular_error(est, normals, mask)


def conditioning(normals, mask):
    """Error against cond(L) as the light azimuths collapse toward one value."""
    albedo = np.full(normals.shape[:2], 0.7)
    conds, errs = [], []
    for spread in [360, 240, 180, 120, 90, 60, 40, 25, 15, 10]:
        lights = near_coplanar_rig(m=5, azimuth_spread_deg=spread)
        lit = fully_lit_mask(normals, lights, mask)
        images = add_noise(render_lambertian(normals, lights, albedo), NOISE, SEED)
        conds.append(condition_number(lights))
        errs.append(_mae(images, lights, normals, lit))
    return np.array(conds), np.array(errs)


def light_count(normals, mask):
    albedo = np.full(normals.shape[:2], 0.7)
    ms = np.array([3, 5, 10, 20, 40])
    errs = []
    for m in ms:
        lights = cone_rig(int(m))
        lit = fully_lit_mask(normals, lights, mask)
        images = add_noise(render_lambertian(normals, lights, albedo), NOISE, SEED)
        errs.append(_mae(images, lights, normals, lit))
    return ms, np.array(errs)


def noise_level(normals, mask):
    albedo = np.full(normals.shape[:2], 0.7)
    lights = cone_rig(10)
    lit = fully_lit_mask(normals, lights, mask)
    clean = render_lambertian(normals, lights, albedo)
    sigmas = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
    # A distinct seed per level. Reusing one draw would make every measurement
    # a rescaling of the same realization, and the fitted exponent would then
    # be exactly one by construction rather than by measurement.
    errs = [_mae(add_noise(clean, float(s), SEED + i), lights, normals, lit)
            for i, s in enumerate(sigmas)]
    return sigmas, np.array(errs)


def shadow_handling(normals, mask):
    """
    Compared on the full mask rather than the fully lit subset, since that is
    the only region where shadowed observations exist at all.

    Swept across c rather than sampled at two points. Masking helps a
    Lambertian surface and helps most of the Minnaert range, but the sign
    reverses at low c, so a two point comparison would support whichever
    conclusion the two points happened to fall on.
    """
    lights = cone_rig(10)
    cs = np.array([1.0, 0.95, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.25, 0.0])
    naive, masked = [], []
    for c in cs:
        images = render_minnaert(normals, lights, float(c))
        naive.append(_mae(images, lights, normals, mask))
        masked.append(_mae(images, lights, normals, mask, woodham_shadow_aware))

    naive, masked = np.array(naive), np.array(masked)
    diff = masked - naive

    # Highest c at which masking stops helping, interpolated between samples.
    crossover = None
    sign = diff > 0
    for i in range(len(cs) - 1):
        if sign[i] != sign[i + 1]:
            t = -diff[i] / (diff[i + 1] - diff[i])
            crossover = float(cs[i] + t * (cs[i + 1] - cs[i]))
            break

    return {"c": cs.tolist(), "naive_deg": naive.tolist(),
            "masked_deg": masked.tolist(), "crossover_c": crossover}


def rig_dependence(normals, mask, c=0.5):
    """Spread of the simplified Hapke error across plausible rigs."""
    vals = []
    for m in [3, 5, 10, 20]:
        for slant in [20.0, 35.0, 50.0]:
            lights = cone_rig(m, slant)
            lit = fully_lit_mask(normals, lights, mask)
            if lit.sum() < 100:
                continue
            vals.append({
                "m": m, "slant_deg": slant,
                "cond_L": condition_number(lights),
                "mae_deg": _mae(render_minnaert(normals, lights, c), lights, normals, lit),
            })
    return vals


def _tilt_stats(normals, mask):
    """Distribution of surface tilt away from the viewing axis, in degrees."""
    t = np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0)))[mask]
    return {"median_deg": float(np.median(t)), "p90_deg": float(np.percentile(t, 90))}


def second_geometry(amplitude=0.25):
    """
    The exponent sweep on a bump field rather than a sphere.

    Tilt statistics are reported alongside, because error magnitude depends on
    how wide a range of orientations the surface presents. Without them a
    flatter surface looks like evidence that the effect is sphere specific,
    when it only shows that the surface spans fewer orientations.
    """
    normals, mask = bump_normals(RES, amplitude=amplitude)
    lights = cone_rig(10)
    lit = fully_lit_mask(normals, lights, mask)
    cs = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    errs = [_mae(render_minnaert(normals, lights, float(c)), lights, normals, lit) for c in cs]
    return cs, np.array(errs), int(lit.sum()), _tilt_stats(normals, lit)


def _plot(cond, lights_n, noise, path):
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH_IN, 1.9))

    # All three panels are log log so the plotted slopes correspond to the
    # fitted exponents and the correlation reported alongside them.
    axes[0].plot(cond[0], cond[1], "o-", color="tab:blue", ms=3)
    axes[0].set_xlabel(r"$\mathrm{cond}(L)$")
    axes[0].set_ylabel("angular error (deg)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    axes[1].plot(lights_n[0], lights_n[1], "o-", color="tab:green", ms=3)
    axes[1].set_xlabel("number of lights $m$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    axes[2].plot(noise[0], noise[1], "o-", color="tab:red", ms=3)
    axes[2].set_xlabel(r"noise $\sigma$")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def run():
    normals, mask = sphere_normals(RES)

    cond = conditioning(normals, mask)
    lights_n = light_count(normals, mask)
    noise = noise_level(normals, mask)
    shadows = shadow_handling(normals, mask)
    rigs = rig_dependence(normals, mask)
    geo_c, geo_err, geo_px, geo_tilt = second_geometry()

    sphere_lit = fully_lit_mask(normals, cone_rig(10), mask)
    sphere_tilt = _tilt_stats(normals, sphere_lit)

    # Fitted power law exponents. Expectation is about -0.5 for light count and
    # about +1 for noise in the small noise regime.
    m_slope = float(np.polyfit(np.log(lights_n[0]), np.log(lights_n[1]), 1)[0])
    small = noise[0] <= 0.01
    n_slope = float(np.polyfit(np.log(noise[0][small]), np.log(noise[1][small]), 1)[0])
    cond_corr = float(np.corrcoef(np.log(cond[0]), np.log(cond[1]))[0, 1])

    mae_rigs = [r["mae_deg"] for r in rigs]
    payload = {
        "seed": SEED, "noise_sigma": NOISE, "resolution": RES,
        "conditioning": {"cond_L": cond[0].tolist(), "mae_deg": cond[1].tolist(),
                         "log_log_correlation": cond_corr},
        "light_count": {"m": lights_n[0].tolist(), "mae_deg": lights_n[1].tolist(),
                        "fitted_exponent": m_slope},
        "noise": {"sigma": noise[0].tolist(), "mae_deg": noise[1].tolist(),
                  "fitted_exponent_small_noise": n_slope},
        "shadow_handling": shadows,
        "rig_dependence": {"runs": rigs, "min_deg": float(min(mae_rigs)),
                           "max_deg": float(max(mae_rigs)), "n_rigs": len(rigs)},
        "second_geometry": {"c": geo_c.tolist(), "mae_deg": geo_err.tolist(),
                            "n_pixels": geo_px, "tilt": geo_tilt,
                            "sphere_tilt": sphere_tilt},
    }

    path = figure_path("fig4_ablations.png")
    _plot(cond, lights_n, noise, path)
    save_section("ablations", payload)

    print(f"conditioning : cond(L) {cond[0].min():.2f} to {cond[0].max():.1f}, "
          f"error {cond[1].min():.3f} to {cond[1].max():.3f} deg, log log r = {cond_corr:.4f}")
    print(f"light count  : fitted exponent {m_slope:+.3f} (expected about -0.5)")
    print(f"noise        : fitted exponent {n_slope:+.3f} (expected about +1)")
    print(f"shadows      : {shadows}")
    print(f"rig spread   : simplified Hapke error {min(mae_rigs):.2f} to {max(mae_rigs):.2f} deg "
          f"over {len(rigs)} rigs")
    print(f"bump surface : " + ", ".join(f"c={c:.2f} {e:.2f}" for c, e in zip(geo_c, geo_err)))
    print(f"  median tilt: bump {geo_tilt['median_deg']:.1f} deg, "
          f"sphere {sphere_tilt['median_deg']:.1f} deg")
    print(f"figure       : {path}")
    return payload


if __name__ == "__main__":
    run()
