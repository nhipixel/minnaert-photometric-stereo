"""
Gate 3. DiLiGenT baseline on real objects with ground truth normals.

These tests skip when the benchmark is not present, so a clean clone still runs
green on the synthetic results alone.

The value of this gate is that it compares against numbers this project did not
produce. A ball error far from the published 4.10 means a convention error in
the light directions, the intensity normalization, or the normal coordinate
frame. It does not mean the method is wrong.
"""
import gc

import numpy as np
import pytest

from diligent import (
    OBJECTS,
    PUBLISHED_AVERAGE,
    PUBLISHED_BASELINE,
    is_available,
    load_object,
    load_official_l2,
)
from metrics import angular_error_deg, summarize
from solver import woodham_lstsq

pytestmark = pytest.mark.skipif(
    not is_available(), reason="DiLiGenT not downloaded, see README"
)


def _mae(name):
    obj = load_object(name)
    est, _ = woodham_lstsq(obj.images, obj.lights, obj.mask)
    value = summarize(angular_error_deg(est, obj.normals_gt, obj.mask), obj.mask)["mean_deg"]
    # Stacks are a few hundred megabytes each and this helper runs dozens of
    # times across the suite, so the reference is dropped before returning.
    del obj, est
    gc.collect()
    return value


def test_object_shapes_are_consistent():
    obj = load_object("ball")
    h, w, m = obj.images.shape
    assert obj.lights.shape == (m, 3)
    assert obj.mask.shape == (h, w)
    assert obj.normals_gt.shape == (h, w, 3)
    assert m == 96


@pytest.mark.parametrize("name", OBJECTS)
def test_ground_truth_normals_are_unit_or_degenerate(name):
    """
    Ground truth normals are unit length except for a small set that are
    exactly zero and therefore carry no direction. Testing only one object hid
    this: nine are clean and pot2 is not. The degenerate pixels are counted
    here so their number is a checked fact rather than an assumption.
    """
    obj = load_object(name)
    lengths = np.linalg.norm(obj.normals_gt[obj.mask], axis=-1)
    degenerate = lengths < 1e-9
    assert np.abs(lengths[~degenerate] - 1.0).max() < 1e-3
    expected = 73 if name == "pot2" else 0
    assert int(degenerate.sum()) == expected, int(degenerate.sum())


def test_light_directions_are_unit_length():
    obj = load_object("ball")
    assert np.abs(np.linalg.norm(obj.lights, axis=1) - 1.0).max() < 1e-6


@pytest.mark.parametrize("name", OBJECTS)
def test_matches_the_shipped_official_baseline_normals(name):
    """
    The strongest check in the project. Compares the estimate against the
    dataset's own precomputed baseline pixel by pixel, not just in aggregate.

    Reproducing this required clipping the grey conversion at 1, since the
    reference clamps its luma output to the unit interval. Without that,
    saturated specular pixels disagree by up to 16 degrees.
    """
    obj = load_object(name)
    est, _ = woodham_lstsq(obj.images, obj.lights, obj.mask)
    official = load_official_l2(name)
    diff = summarize(angular_error_deg(est, official, obj.mask), obj.mask)
    assert diff["mean_deg"] < 0.01, diff


@pytest.mark.parametrize("name", OBJECTS)
def test_matches_the_tabulated_literature_baseline(name):
    """
    Aggregate check against the commonly tabulated numbers.

    pot2 gets a wider tolerance for a documented reason. Its 73 degenerate
    ground truth pixels are scored as 90 degrees by the reference metric and
    excluded here, which accounts for the whole difference. The tabulated value
    is correct; the two conventions simply disagree on undefined pixels.
    """
    tolerance = 0.25 if name == "pot2" else 0.02
    assert abs(_mae(name) - PUBLISHED_BASELINE[name]) < tolerance


def test_degenerate_pixels_explain_the_pot2_difference():
    """
    Scoring the 73 undefined pixels as 90 degrees, which is what arccos of a
    zero dot product returns, should recover the tabulated value exactly. This
    pins the cause rather than leaving it as an unexplained tolerance.
    """
    obj = load_object("pot2")
    est, _ = woodham_lstsq(obj.images, obj.lights, obj.mask)
    err = angular_error_deg(est, obj.normals_gt, obj.mask)

    scored = np.isfinite(err) & obj.mask
    n_degenerate = int(obj.mask.sum() - scored.sum())
    assert n_degenerate == 73

    with_reference_convention = (
        np.nansum(err[scored]) + n_degenerate * 90.0
    ) / int(obj.mask.sum())
    assert abs(with_reference_convention - PUBLISHED_BASELINE["pot2"]) < 0.01


def test_average_matches_the_published_average():
    values = [_mae(n) for n in OBJECTS]
    assert abs(float(np.mean(values)) - PUBLISHED_AVERAGE) < 0.05


def test_diffuse_objects_beat_specular_ones():
    assert _mae("ball") < _mae("cat") < _mae("pot2") < _mae("harvest")
