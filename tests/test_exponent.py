"""
Gate for the effective exponent fit and the implied correction.

The synthetic half must be exact, since the log linear relation holds by
construction there. The real half asserts the study's load bearing findings:
the oracle bound helps everywhere, the fitted exponents sit above the Minnaert
domain, and the naive alternation does not work.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from brdf import render_minnaert
from diligent import OBJECTS, is_available, load_object
from exponent import apply_exponent_correction, fit_exponent_map
from geometry import sphere_normals
from lights import cone_rig
from metrics import mean_angular_error
from solver import woodham_lstsq, woodham_trimmed
from theory import fully_lit_mask

RES = 96


@pytest.fixture(scope="module")
def sphere():
    normals, mask = sphere_normals(RES)
    lights = cone_rig(10)
    return normals, mask, lights, fully_lit_mask(normals, lights, mask)


@pytest.mark.parametrize("c", [0.25, 0.5, 0.75, 1.0])
def test_fit_recovers_a_known_exponent_exactly(sphere, c):
    """On noiseless Minnaert data the log log relation is exactly linear."""
    normals, mask, lights, lit = sphere
    images = render_minnaert(normals, lights, c)
    c_map = fit_exponent_map(images, normals, lights)
    assert np.abs(c_map[lit] - c).max() < 1e-8


@pytest.mark.parametrize("c", [0.25, 0.5, 0.75])
def test_correction_with_the_true_exponent_is_exact(sphere, c):
    """
    Raising each observation to 1/c turns the stack Lambertian by identity,
    so the unchanged solver must recover normals to machine precision.
    """
    normals, mask, lights, lit = sphere
    images = render_minnaert(normals, lights, c)

    est, _ = woodham_lstsq(apply_exponent_correction(images, c), lights, mask)
    assert mean_angular_error(est, normals, lit) < 1e-9

    c_map = np.full(mask.shape, c)
    est, _ = woodham_lstsq(apply_exponent_correction(images, c_map), lights, mask)
    assert mean_angular_error(est, normals, lit) < 1e-9


def test_trimmed_solver_is_exact_on_clean_lambertian_data(sphere):
    """Dropping valid observations from consistent data must change nothing."""
    normals, mask, lights, lit = sphere
    images = render_minnaert(normals, lights, 1.0)
    est, _ = woodham_trimmed(images, lights, mask)
    assert mean_angular_error(est, normals, lit) < 1e-6


needs_data = pytest.mark.skipif(not is_available(), reason="DiLiGenT not downloaded")


@pytest.fixture(scope="module")
def real_pair():
    return {name: load_object(name) for name in ("ball", "harvest")}


@needs_data
@pytest.mark.parametrize("name", ["ball", "harvest"])
def test_real_exponents_sit_above_the_minnaert_domain(real_pair, name):
    """
    The fitted exponent exceeds one on real objects, glossy materials brighten
    faster than the cosine. This is a finding the report states, so it is
    pinned here rather than left as prose.
    """
    obj = real_pair[name]
    c_map = fit_exponent_map(obj.images, obj.normals_gt, obj.lights)
    assert float(np.median(c_map[obj.mask])) > 1.0


@needs_data
def test_oracle_correction_beats_the_baseline_on_every_object():
    """
    The oracle bound must help everywhere or it is not a bound worth quoting.
    Averages alone could hide an object it damages.
    """
    improvements = []
    for name in OBJECTS:
        obj = load_object(name)
        base_est, _ = woodham_lstsq(obj.images, obj.lights, obj.mask)
        base = mean_angular_error(base_est, obj.normals_gt, obj.mask)

        c_map = fit_exponent_map(obj.images, obj.normals_gt, obj.lights)
        est, _ = woodham_lstsq(
            apply_exponent_correction(obj.images, np.where(obj.mask, c_map, 1.0)),
            obj.lights, obj.mask,
        )
        corrected = mean_angular_error(est, obj.normals_gt, obj.mask)
        assert corrected < base, (name, corrected, base)
        improvements.append(base - corrected)
    assert float(np.mean(improvements)) > 3.0


@needs_data
def test_naive_alternation_hurts_where_the_oracle_helps(real_pair):
    """
    Fitting the exponent against the current estimate reinforces that
    estimate's own bias. The divergence is the study's negative result and is
    asserted so it cannot be quietly dropped.
    """
    obj = real_pair["ball"]
    base_est, _ = woodham_lstsq(obj.images, obj.lights, obj.mask)
    base = mean_angular_error(base_est, obj.normals_gt, obj.mask)

    cur = base_est
    for _ in range(3):
        c_iter = fit_exponent_map(obj.images, cur, obj.lights)
        cur, _ = woodham_lstsq(
            apply_exponent_correction(obj.images, c_iter), obj.lights, obj.mask
        )
    assert mean_angular_error(cur, obj.normals_gt, obj.mask) > base
