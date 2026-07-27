"""
Reflectance models and image formation.

The Minnaert model (Minnaert 1941) is used in its normalized form:

    f_r = ((c+1) / 2pi) k (cos_i cos_r)^(c-1),        0 <= c <= 1
    L   = ((c+1) / 2pi) k E0 cos_i^c cos_r^(c-1)

with c = 1 giving Lambertian and c = 0.5 giving the simplified Hapke model.

Two independent implementations of the same Minnaert radiance are provided.
render_minnaert evaluates the BRDF and the foreshortening term separately,
while render_minnaert_powerlaw uses the reduction derived in the report. They
must agree, and that agreement is the numerical proof of the reduction.
"""
import numpy as np

from lights import view_direction


def _cos_incident(normals, lights):
    """
    Cosine of the incident angle for every pixel and every light.

    Returns shape (H, W, m). Negative values mean the surface faces away from
    the light, which is attached shadow, so they are clamped to zero.
    """
    cos_i = normals @ lights.T
    return np.clip(cos_i, 0.0, None)


def _shadow_mask(normals, lights):
    """True where the light is above the local horizon and radiance is valid."""
    return (normals @ lights.T) > 0.0


def render_lambertian(normals, lights, albedo):
    """Lambertian image stack, I_j = albedo * max(0, n dot s_j)."""
    return _cos_incident(normals, lights) * albedo[..., None]


def render_minnaert(normals, lights, c, k=1.0, E0=1.0, view=None):
    """
    Minnaert radiance evaluated directly from the BRDF times the
    foreshortening term. This is the reference path.
    """
    if view is None:
        view = view_direction()

    cos_i = _cos_incident(normals, lights)
    cos_r = np.clip(normals @ view, 0.0, None)[..., None]
    lit = _shadow_mask(normals, lights)

    out = np.zeros_like(cos_i)
    prefactor = (c + 1.0) / (2.0 * np.pi) * k
    with np.errstate(divide="ignore", invalid="ignore"):
        brdf = prefactor * np.power(cos_i * cos_r, c - 1.0)
        out = np.where(lit, brdf * E0 * cos_i, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def render_minnaert_powerlaw(normals, lights, c, k=1.0, E0=1.0, view=None):
    """
    Same radiance written using the reduction. Under orthographic viewing
    cos_r equals n_z and does not depend on the light index, so the whole
    view dependent factor collapses into a per pixel constant A and the
    image stack becomes a power law on Lambertian shading:

        I_j = A (n dot s_j)^c,   A = ((c+1)/2pi) k E0 n_z^(c-1)
    """
    if view is None:
        view = view_direction()

    cos_i = _cos_incident(normals, lights)
    cos_r = np.clip(normals @ view, 0.0, None)
    lit = _shadow_mask(normals, lights)

    with np.errstate(divide="ignore", invalid="ignore"):
        A = (c + 1.0) / (2.0 * np.pi) * k * E0 * np.power(cos_r, c - 1.0)
        out = np.where(lit, A[..., None] * np.power(cos_i, c), 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def minnaert_A(normals, c, k=1.0, E0=1.0, view=None):
    """The per pixel constant A. Exposed so the albedo bias can be checked."""
    if view is None:
        view = view_direction()
    cos_r = np.clip(normals @ view, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (c + 1.0) / (2.0 * np.pi) * k * E0 * np.power(cos_r, c - 1.0)
    return np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)


def render_simplified_hapke(normals, lights, L0=1.0, view=None):
    """
    Simplified Hapke model in the form

        L_r = L0 sqrt((n dot s) / (n dot v)).

    This is a separate code path from render_minnaert with c = 0.5 and is used
    to cross check the claim that simplified Hapke is Minnaert at c = 0.5.

    Note the full Hapke model, which adds multiple scattering, shadowing and
    porosity, is not a special case of Minnaert. Only the simplified form is.
    """
    if view is None:
        view = view_direction()

    cos_i = _cos_incident(normals, lights)
    cos_r = np.clip(normals @ view, 0.0, None)[..., None]
    lit = _shadow_mask(normals, lights)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(lit, L0 * np.sqrt(cos_i / cos_r), 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def add_noise(images, sigma, seed=0):
    """Additive zero mean Gaussian sensor noise with a recorded seed."""
    if sigma <= 0.0:
        return images.copy()
    rng = np.random.default_rng(seed)
    return images + rng.normal(0.0, sigma, size=images.shape)
