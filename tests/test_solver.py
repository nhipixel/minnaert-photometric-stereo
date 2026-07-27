"""
Gate 1. Exact recovery of the Lambertian forward model.

With noiseless Lambertian data the measurements lie exactly in the column
space of L, so least squares must reproduce the normals and the albedo to
machine precision. Anything larger indicates a convention bug in the light
directions, the matrix orientation, or the mask, and every later result would
inherit it.
"""
import numpy as np
import pytest

from brdf import render_lambertian
from geometry import checker_albedo, sphere_normals
from lights import condition_number, cone_rig
from metrics import angular_error_deg, summarize
from solver import woodham_lstsq

RES = 128
EXACT = 1e-10


@pytest.mark.parametrize("m", [3, 5, 10])
def test_exact_recovery_uniform_albedo(m):
    normals, mask = sphere_normals(RES)
    lights = cone_rig(m)
    albedo = np.full((RES, RES), 0.7)

    images = render_lambertian(normals, lights, albedo)
    est_n, est_rho = woodham_lstsq(images, lights, mask)

    # Only fully lit pixels satisfy the linear model. A pixel in attached
    # shadow has a clamped observation that the model does not describe.
    lit = np.all((normals @ lights.T) > 0.0, axis=-1) & mask

    err = angular_error_deg(est_n, normals, lit)
    stats = summarize(err, lit)
    assert stats["max_deg"] < EXACT, stats

    rho_err = np.abs(est_rho - albedo)[lit].max()
    assert rho_err < 1e-12, rho_err


def test_albedo_and_orientation_decouple():
    """A spatially varying albedo must not leak into the normal estimate."""
    normals, mask = sphere_normals(RES)
    lights = cone_rig(10)
    albedo = checker_albedo(RES)

    images = render_lambertian(normals, lights, albedo)
    est_n, est_rho = woodham_lstsq(images, lights, mask)
    lit = np.all((normals @ lights.T) > 0.0, axis=-1) & mask

    assert summarize(angular_error_deg(est_n, normals, lit), lit)["max_deg"] < EXACT
    assert np.abs(est_rho - albedo)[lit].max() < 1e-12


def test_estimated_normals_are_unit_length():
    normals, mask = sphere_normals(RES)
    lights = cone_rig(5)
    images = render_lambertian(normals, lights, np.full((RES, RES), 0.7))
    est_n, _ = woodham_lstsq(images, lights, mask)

    lit = np.all((normals @ lights.T) > 0.0, axis=-1) & mask
    lengths = np.linalg.norm(est_n[lit], axis=-1)
    assert np.abs(lengths - 1.0).max() < 1e-12


def test_ground_truth_sphere_normals_are_analytic():
    normals, mask = sphere_normals(RES)
    lengths = np.linalg.norm(normals[mask], axis=-1)
    assert np.abs(lengths - 1.0).max() < 1e-12
    # Trimming the rim keeps n_z bounded away from zero.
    assert normals[mask][:, 2].min() > 0.0


def test_default_rig_is_well_conditioned():
    assert condition_number(cone_rig(3)) < 10.0
    assert condition_number(cone_rig(10)) < 10.0
