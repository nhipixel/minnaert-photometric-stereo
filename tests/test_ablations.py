"""
Gate 5. Each ablation has an expectation stated before the run.

An ablation that merely produces a plot proves nothing. These assert the
quantitative behaviour that theory predicts, so a regression in the solver or
the rendering shows up as a failed expectation rather than a slightly different
looking curve.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from exp4_ablations import (
    conditioning,
    light_count,
    noise_level,
    rig_dependence,
    second_geometry,
    shadow_handling,
)
from geometry import bump_normals, sphere_normals

RES = 96


@pytest.fixture(scope="module")
def sphere():
    return sphere_normals(RES)


def test_error_grows_monotonically_with_conditioning(sphere):
    """
    Squeezing the light azimuths toward a common value raises cond(L) and
    amplifies noise with it, which is the standard argument for requiring light
    directions that are not coplanar in azimuth.

    Growth is asserted as monotone rather than proportional. The fitted log log
    slope is well below one, and the most degenerate rigs saturate, so claiming
    proportionality would overstate what the data shows.
    """
    conds, errs = conditioning(*sphere)
    assert conds[0] < 10 and conds[-1] > 100
    assert np.all(np.diff(errs) > 0)
    assert errs[-1] > 20 * errs[0]
    r = np.corrcoef(np.log(conds), np.log(errs))[0, 1]
    assert r > 0.95, r
    slope = np.polyfit(np.log(conds), np.log(errs), 1)[0]
    assert 0.5 < slope < 1.0, slope


def test_error_falls_as_inverse_root_of_light_count(sphere):
    ms, errs = light_count(*sphere)
    slope = np.polyfit(np.log(ms), np.log(errs), 1)[0]
    assert -0.6 < slope < -0.4, slope


def test_error_grows_linearly_with_noise(sphere):
    sigmas, errs = noise_level(*sphere)
    small = sigmas <= 0.01
    slope = np.polyfit(np.log(sigmas[small]), np.log(errs[small]), 1)[0]
    assert 0.9 < slope < 1.1, slope


def test_shadow_masking_is_exact_for_lambertian(sphere):
    """
    Dropping observations below the local horizon removes the only error source
    in the Lambertian case, so recovery becomes exact on the full mask.
    """
    out = shadow_handling(*sphere)
    assert out["naive_deg"][0] > 0.5
    assert out["masked_deg"][0] < 1e-10


def test_shadow_masking_reverses_sign_partway_down_the_range(sphere):
    """
    Masking helps for most of the Minnaert range and hurts at low c, so the
    benefit is not a general property of non Lambertian surfaces.

    The mechanism is that masking cannot remove a systematic reflectance bias,
    only outliers. Retaining the clamped rows happens to shrink the estimate in
    a direction that partly cancels that bias, and once the bias is large
    enough that accidental cancellation is worth more than the bad rows cost.
    Asserting the crossover keeps the claim tied to where it actually holds.
    """
    out = shadow_handling(*sphere)
    naive = np.array(out["naive_deg"])
    masked = np.array(out["masked_deg"])
    cs = np.array(out["c"])

    assert np.all(masked[cs >= 0.75] < naive[cs >= 0.75])
    assert np.all(masked[cs <= 0.5] > naive[cs <= 0.5])
    assert 0.6 < out["crossover_c"] < 0.8, out["crossover_c"]


def test_headline_error_is_rig_dependent(sphere):
    """
    The simplified Hapke error is not a universal constant. Its spread across
    plausible rigs is measured so the reported figure is always qualified.
    """
    runs = rig_dependence(*sphere)
    vals = [r["mae_deg"] for r in runs]
    assert len(runs) >= 8
    assert max(vals) - min(vals) > 2.0
    assert all(5.0 < v < 25.0 for v in vals)


def test_bump_surface_reproduces_the_shape_of_the_curve():
    """
    The curve must not be a property of the sphere. On a bump field the shape
    is preserved: exact at c = 1 and monotone as c falls. The magnitude differs
    because the surface presents a narrower range of orientations, which is why
    the tilt statistics are reported next to it.
    """
    cs, errs, n_px, tilt = second_geometry()
    assert n_px > 1000
    assert errs[0] < 1e-10
    assert np.all(np.diff(errs) > 0)
    assert tilt["median_deg"] > 0.0


def test_bump_surface_normals_are_unit_and_face_the_camera():
    normals, mask = bump_normals(RES)
    assert np.abs(np.linalg.norm(normals[mask], axis=-1) - 1.0).max() < 1e-12
    assert normals[..., 2].min() > 0.0
