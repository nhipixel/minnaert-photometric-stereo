"""
Gate 4. The exponent sweep behaves the way the derivation says it must.

Three points on this curve are fixed in advance by theory, not by fitting, and
each is asserted here. The sweep would be a plausible looking plot without
them; with them it is a prediction that either holds or does not.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from brdf import render_minnaert, render_simplified_hapke
from exp3_sweep_c import sweep
from geometry import checker_albedo, sphere_normals
from lights import cone_rig
from metrics import angular_error_deg, summarize
from solver import woodham_lstsq
from theory import collapse_error_deg, fully_lit_mask

RES = 128
COARSE = np.round(np.arange(0.0, 1.0001, 0.05), 4)


@pytest.fixture(scope="module")
def data():
    return sweep(resolution=RES, c_values=COARSE)


def test_lambertian_endpoint_is_exact(data):
    """c = 1 must land at machine precision or the whole curve is suspect."""
    assert data["mean"][-1] < 1e-10


def test_error_grows_monotonically_as_c_falls(data):
    # Arrays run from c = 0 up to c = 1, so error must be nonincreasing in c.
    assert np.all(np.diff(data["mean"]) <= 1e-9)
    assert np.all(np.diff(data["median"]) <= 1e-9)


def test_c_zero_endpoint_matches_the_closed_form_collapse(data):
    assert abs(data["mean"][0] - data["collapse"]) < 0.01


def test_hapke_point_matches_an_independent_render():
    """
    The c = 0.5 value must be reproducible from the simplified Hapke code path,
    which shares no code with the Minnaert renderer.
    """
    normals, mask = sphere_normals(RES)
    lights = cone_rig(10)
    lit = fully_lit_mask(normals, lights, mask)

    via_minnaert, _ = woodham_lstsq(render_minnaert(normals, lights, 0.5), lights, mask)
    via_hapke, _ = woodham_lstsq(render_simplified_hapke(normals, lights), lights, mask)

    a = summarize(angular_error_deg(via_minnaert, normals, lit), lit)["mean_deg"]
    b = summarize(angular_error_deg(via_hapke, normals, lit), lit)["mean_deg"]
    assert abs(a - b) < 1e-10


def test_first_order_law_holds_near_lambertian_and_is_reported_where_it_fails(data):
    """
    The prediction is first order in 1 - c, so it must be accurate close to
    Lambertian and must visibly understate the error far from it. Both halves
    are asserted, because a prediction that silently drifts is worse than one
    with a stated range of validity.
    """
    c, mean, pred = data["c"], data["mean"], data["predicted"]

    near = c >= 0.9
    rel = np.abs(pred[near] - mean[near]) / np.maximum(mean[near], 1e-12)
    assert rel.max() < 0.10

    far = c <= 0.3
    assert np.all(pred[far] < mean[far])


def test_curve_is_invariant_to_the_albedo_field():
    """
    P1 again, now at the level of the whole curve. A per pixel albedo scales
    every observation at that pixel equally, so it cannot move a normal.
    """
    normals, mask = sphere_normals(RES)
    lights = cone_rig(10)
    lit = fully_lit_mask(normals, lights, mask)
    albedo = checker_albedo(RES)

    for c in [0.75, 0.5, 0.25]:
        images = render_minnaert(normals, lights, c)
        plain, _ = woodham_lstsq(images, lights, mask)
        scaled, _ = woodham_lstsq(images * albedo[..., None], lights, mask)
        assert summarize(angular_error_deg(plain, scaled, lit), lit)["max_deg"] < 1e-10


def test_curve_is_stable_under_resolution_change():
    coarse = sweep(resolution=96, c_values=np.array([0.5]))
    fine = sweep(resolution=192, c_values=np.array([0.5]))
    assert abs(coarse["mean"][0] - fine["mean"][0]) < 0.05


def test_collapse_direction_is_independent_of_the_true_normal():
    """At c = 0 every fully lit pixel returns the same estimate."""
    normals, mask = sphere_normals(RES)
    lights = cone_rig(10)
    lit = fully_lit_mask(normals, lights, mask)

    est, _ = woodham_lstsq(render_minnaert(normals, lights, 0.0), lights, mask)
    spread = est[lit] - est[lit].mean(axis=0)
    assert np.abs(spread).max() < 1e-10

    measured = summarize(angular_error_deg(est, normals, lit), lit)["mean_deg"]
    predicted = float(np.nanmean(collapse_error_deg(normals, lights, lit)))
    assert abs(measured - predicted) < 1e-6
