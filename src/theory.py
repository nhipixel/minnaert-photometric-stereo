"""
Closed form predictions for the error of a Lambertian solver on a Minnaert
surface. These are derived in the report and verified numerically by the
tests, so nothing here is fitted to the measurements.
"""
import numpy as np


def fully_lit_mask(normals, lights, mask=None):
    """
    Pixels where every light is above the local horizon.

    The reduction I_j = A (n dot s_j)^c only holds where no observation is
    clamped by attached shadow, so the closed form predictions are stated on
    this subset.
    """
    lit = (normals @ lights.T) > 0.0
    allsee = np.all(lit, axis=-1)
    return allsee if mask is None else (allsee & mask)


def first_order_error_deg(normals, lights, c, mask=None):
    """
    First order angular error in the Minnaert exponent.

    Writing c = 1 - eps and expanding (n dot s)^c about eps = 0 gives a
    radiance perturbation dI_j = -eps A u_j with u_j = (n dot s_j) ln(n dot s_j).
    Propagating through g = pinv(L) I and keeping the part perpendicular to
    the true normal, the per pixel angle is

        theta ~= eps * norm(P_perp pinv(L) u),   P_perp = I - n n^T

    The per pixel constant A cancels, so the prediction does not depend on
    albedo, on irradiance, or on n_z.
    """
    eps = 1.0 - c
    cos_i = normals @ lights.T
    lit = cos_i > 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(lit, cos_i * np.log(np.where(lit, cos_i, 1.0)), 0.0)
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

    v = u @ np.linalg.pinv(lights).T
    v_perp = v - np.sum(v * normals, axis=-1, keepdims=True) * normals
    theta = eps * np.linalg.norm(v_perp, axis=-1)

    out = np.degrees(theta)
    if mask is not None:
        out = np.where(mask, out, np.nan)
    return out


def collapse_direction(lights):
    """
    Direction the estimated normal field collapses to at c = 0.

    At c = 0 every unshadowed observation equals the same per pixel constant,
    so I is A times the all ones vector and g = A pinv(L) 1. The direction of
    g is then identical at every fully lit pixel, independent of the true
    normal.
    """
    ones = np.ones(lights.shape[0])
    g = np.linalg.pinv(lights) @ ones
    return g / np.linalg.norm(g)


def collapse_error_deg(normals, lights, mask=None):
    """Angle between each true normal and the c = 0 collapse direction."""
    d = collapse_direction(lights)
    dot = np.clip(normals @ d, -1.0, 1.0)
    err = np.degrees(np.arccos(dot))
    if mask is not None:
        err = np.where(mask, err, np.nan)
    return err
