"""
Gate 3. DiLiGenT baseline on real objects with ground truth normals.

These tests skip when the benchmark is not present, so a clean clone still runs
green on the synthetic results alone.

The value of this gate is that it compares against numbers this project did not
produce. A ball error far from the published 4.10 means a convention error in
the light directions, the intensity normalization, or the normal coordinate
frame. It does not mean the method is wrong.
"""
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
    err = angular_error_deg(est, obj.normals_gt, obj.mask)
    return summarize(err, obj.mask)["mean_deg"]


def test_object_shapes_are_consistent():
    obj = load_object("ball")
    h, w, m = obj.images.shape
    assert obj.lights.shape == (m, 3)
    assert obj.mask.shape == (h, w)
    assert obj.normals_gt.shape == (h, w, 3)
    assert m == 96


def test_ground_truth_normals_are_unit_length_inside_the_mask():
    obj = load_object("ball")
    lengths = np.linalg.norm(obj.normals_gt[obj.mask], axis=-1)
    assert np.abs(lengths - 1.0).max() < 1e-3


def test_light_directions_are_unit_length():
    obj = load_object("ball")
    assert np.abs(np.linalg.norm(obj.lights, axis=1) - 1.0).max() < 1e-6


@pytest.mark.parametrize("name", OBJECTS)
def test_matches_the_shipped_official_baseline_normals(name):
    """
    The strongest check in the project. Compares the estimate against the
    dataset's own precomputed baseline pixel by pixel, not just in aggregate.

    Reproducing this required clipping the grey conversion at 1, since the
    reference uses Matlab rgb2gray, which clamps double output. Without that,
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
    Weaker aggregate check against the commonly tabulated numbers.

    pot2 gets a wider tolerance. Its shipped baseline artifact yields 14.46
    while the literature tabulates 14.65, and the estimate here matches the
    artifact to 0.0002 degrees. The discrepancy is in the table, not the solver.
    """
    tolerance = 0.25 if name == "pot2" else 0.02
    assert abs(_mae(name) - PUBLISHED_BASELINE[name]) < tolerance


def test_average_matches_the_published_average():
    values = [_mae(n) for n in OBJECTS]
    assert abs(float(np.mean(values)) - PUBLISHED_AVERAGE) < 0.05


def test_diffuse_objects_beat_specular_ones():
    assert _mae("ball") < _mae("cat") < _mae("pot2") < _mae("harvest")
