"""
Effective Minnaert exponent estimation and the implied correction.

The reduction I_j = A (n dot s_j)^c is linear in log space,

    log I_j = log A + c log(n dot s_j)

so with known normals the exponent is the slope of a per pixel least squares
line. On synthetic Minnaert data the relation is exactly linear and the fit is
exact. On real data the fit is a diagnostic of how far a material departs from
Lambertian, not a method input, since it uses the ground truth normals.

The correction follows from the same identity. Raising each observation to
1/c gives I_j^(1/c) = A^(1/c) (n dot s_j), which is exactly Lambertian with a
different albedo, so the unmodified solver applies.
"""
import numpy as np

EPS = 1e-6


def fit_exponent_map(images, normals, lights, min_cos=0.1, min_obs=4):
    """
    Per pixel slope of log intensity against log shading.

    Observations are used only where the surface faces the light by more than
    min_cos and the measurement is positive, since shadowed or clipped values
    do not follow the power law. Pixels with fewer than min_obs usable
    observations keep c = 1, the Lambertian default.
    """
    cos_i = normals @ lights.T
    ok = (cos_i > min_cos) & (images > EPS)

    x = np.where(ok, np.log(np.clip(cos_i, EPS, None)), 0.0)
    y = np.where(ok, np.log(np.clip(images, EPS, None)), 0.0)
    n = ok.sum(axis=-1)

    sx, sy = x.sum(-1), y.sum(-1)
    sxx = (x * x).sum(-1)
    sxy = (x * y).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    c = np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=1.0)
    c[n < min_obs] = 1.0
    # The wide clip keeps pathological pixels finite without hiding fits that
    # land outside the Minnaert domain, which are reported, not clamped to 1.
    return np.clip(c, 0.05, 3.0)


def apply_exponent_correction(images, c):
    """
    Raise each observation to 1/c so a Minnaert stack becomes Lambertian.

    Accepts a scalar exponent or a per pixel map. Zero stays zero, so attached
    shadows remain attached shadows.
    """
    c = np.asarray(c, dtype=float)
    if c.ndim == 0:
        power = 1.0 / float(c)
        return np.power(np.clip(images, 0.0, None), power)
    return np.power(np.clip(images, 0.0, None), (1.0 / c)[..., None])
