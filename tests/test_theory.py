"""
Gate 2. The power law reduction and the four predictions that follow from it.

Under orthographic viewing cos_r equals n_z and is the same for every light,
so the Minnaert image stack reduces to

    I_j = A (n dot s_j)^c,    A = ((c+1)/2pi) k E0 n_z^(c-1).

Everything the report claims about where Lambertian photometric stereo breaks
rests on this identity, so it is checked symbolically and numerically before
it is used.
"""
import numpy as np
import pytest
import sympy as sp

from brdf import (
    minnaert_A,
    render_lambertian,
    render_minnaert,
    render_minnaert_powerlaw,
    render_simplified_hapke,
)
from geometry import sphere_normals
from lights import cone_rig
from metrics import angular_error_deg, summarize
from solver import woodham_lstsq
from theory import (
    collapse_direction,
    collapse_error_deg,
    first_order_error_deg,
    fully_lit_mask,
)

RES = 128
C_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def _setup(m=10):
    normals, mask = sphere_normals(RES)
    lights = cone_rig(m)
    lit = fully_lit_mask(normals, lights, mask)
    return normals, mask, lights, lit


def test_symbolic_reduction_matches_the_renderer():
    """
    The symbolic radiance and the implemented renderer must agree.

    Comparing two symbolic expressions to each other would only exercise sympy.
    The symbolic form is lambdified and evaluated against render_minnaert so
    the identity is tied to the code that actually produces the images.
    """
    c, k, E0 = sp.symbols("c k E0", positive=True)
    ci, cr = sp.symbols("cos_i cos_r", positive=True)

    f_r = (c + 1) / (2 * sp.pi) * k * (ci * cr) ** (c - 1)
    radiance = sp.simplify(f_r * E0 * ci)
    assert sp.simplify(radiance - (c + 1) / (2 * sp.pi) * k * E0 * ci**c * cr ** (c - 1)) == 0

    f = sp.lambdify((ci, cr, c, k, E0), radiance, "numpy")

    normals, mask, lights, lit = _setup()
    cos_i = np.clip(normals @ lights.T, 0.0, None)
    cos_r = np.clip(normals[..., 2], 0.0, None)[..., None]
    for c_val in [0.25, 0.5, 0.75, 1.0]:
        # Shadowed pixels have cos_i of zero, where the symbolic form divides
        # by zero for c below one. The renderer clamps them, so they are
        # excluded here rather than compared.
        with np.errstate(divide="ignore", invalid="ignore"):
            symbolic = f(cos_i, np.broadcast_to(cos_r, cos_i.shape), c_val, 1.0, 1.0)
        rendered = render_minnaert(normals, lights, c_val)
        scale = np.abs(rendered[lit]).max()
        assert np.abs(symbolic - rendered)[lit].max() / scale < 1e-12


@pytest.mark.parametrize("c", C_VALUES)
def test_powerlaw_matches_direct_render(c):
    """Two independent implementations of the same radiance must agree."""
    normals, mask, lights, lit = _setup()

    direct = render_minnaert(normals, lights, c)
    reduced = render_minnaert_powerlaw(normals, lights, c)

    scale = np.abs(direct[lit]).max()
    assert np.abs(direct - reduced)[lit].max() / scale < 1e-12


def test_c_equals_one_is_lambertian():
    normals, mask, lights, lit = _setup()
    minnaert = render_minnaert(normals, lights, 1.0)
    # The Minnaert prefactor at c = 1 is (1+1)/(2 pi) = 1/pi.
    lambertian = render_lambertian(normals, lights, np.full((RES, RES), 1.0 / np.pi))
    assert np.abs(minnaert - lambertian)[lit].max() < 1e-14


def test_c_equals_half_is_simplified_hapke():
    """
    Simplified Hapke is Minnaert at c = 0.5, so the two renders differ only by
    a constant prefactor. The full Hapke model is not a special case of
    Minnaert and is not tested here.
    """
    normals, mask, lights, lit = _setup()
    minnaert = render_minnaert(normals, lights, 0.5)
    hapke = render_simplified_hapke(normals, lights)

    ratio = minnaert[lit] / hapke[lit]
    assert np.ptp(ratio) / np.mean(ratio) < 1e-12
    assert np.isclose(np.mean(ratio), 1.5 / (2.0 * np.pi))


@pytest.mark.parametrize("c", [0.25, 0.5, 0.75])
def test_p1_normals_are_invariant_to_the_per_pixel_constant(c):
    """
    P1. A scales every observation at a pixel equally, so it cancels in the
    normal direction. Dropping the n_z factor from A must leave the estimated
    normals bitwise unchanged.
    """
    normals, mask, lights, lit = _setup()

    with_nz = render_minnaert_powerlaw(normals, lights, c)
    cos_i = np.clip(normals @ lights.T, 0.0, None)
    without_nz = (c + 1.0) / (2.0 * np.pi) * np.power(cos_i, c)

    n_a, _ = woodham_lstsq(with_nz, lights, mask)
    n_b, _ = woodham_lstsq(without_nz, lights, mask)

    assert summarize(angular_error_deg(n_a, n_b, lit), lit)["max_deg"] < 1e-10


def test_p1_normals_are_invariant_to_albedo_and_irradiance():
    normals, mask, lights, lit = _setup()
    a = render_minnaert(normals, lights, 0.5, k=1.0, E0=1.0)
    b = render_minnaert(normals, lights, 0.5, k=0.3, E0=7.5)

    n_a, _ = woodham_lstsq(a, lights, mask)
    n_b, _ = woodham_lstsq(b, lights, mask)
    assert summarize(angular_error_deg(n_a, n_b, lit), lit)["max_deg"] < 1e-10


@pytest.mark.parametrize("c", [0.25, 0.5, 0.75])
def test_p2_albedo_carries_the_full_nz_bias(c):
    """
    P2. The same factor that cancels in the normal survives in the albedo.
    Since g = A pinv(L) w with w_j = (n dot s_j)^c, the estimated albedo is
    exactly A times a term that does not involve A, so dividing it out must
    leave the n_z power law behind.
    """
    normals, mask, lights, lit = _setup()

    images = render_minnaert_powerlaw(normals, lights, c)
    _, rho = woodham_lstsq(images, lights, mask)

    w = np.clip(normals @ lights.T, 0.0, None) ** c
    g_unit = w @ np.linalg.pinv(lights).T
    norm_term = np.linalg.norm(g_unit, axis=-1)

    ratio = rho[lit] / norm_term[lit]
    predicted = minnaert_A(normals, c)[lit]
    assert np.abs(ratio - predicted).max() / predicted.max() < 1e-12

    # The residual factor is exactly n_z^(c-1), so a log log fit against n_z
    # must recover the exponent c-1. This states the divergence toward the
    # silhouette without depending on how much of the silhouette is fully lit.
    nz = normals[..., 2][lit]
    slope, _ = np.polyfit(np.log(nz), np.log(ratio), 1)
    assert abs(slope - (c - 1.0)) < 1e-10, slope


def test_p4_normal_field_collapses_at_c_zero():
    """
    P4. At c = 0 every fully lit observation equals A, so g = A pinv(L) 1 and
    the estimated direction is the same at every such pixel regardless of the
    true normal. This is a specific wrong answer, not noise.
    """
    normals, mask, lights, lit = _setup()

    images = render_minnaert_powerlaw(normals, lights, 0.0)
    est_n, _ = woodham_lstsq(images, lights, mask)

    d = collapse_direction(lights)
    spread = angular_error_deg(est_n, np.broadcast_to(d, est_n.shape), lit)
    assert summarize(spread, lit)["max_deg"] < 1e-10

    measured = summarize(angular_error_deg(est_n, normals, lit), lit)["mean_deg"]
    predicted = float(np.nanmean(collapse_error_deg(normals, lights, lit)))
    assert abs(measured - predicted) < 1e-6


@pytest.mark.parametrize("c", [0.99, 0.98])
def test_first_order_law_matches_measurement_near_lambertian(c):
    normals, mask, lights, lit = _setup()

    images = render_minnaert_powerlaw(normals, lights, c)
    est_n, _ = woodham_lstsq(images, lights, mask)

    measured = angular_error_deg(est_n, normals, lit)[lit]
    predicted = first_order_error_deg(normals, lights, c, lit)[lit]

    rel = abs(measured.mean() - predicted.mean()) / measured.mean()
    assert rel < 0.05, (measured.mean(), predicted.mean(), rel)
