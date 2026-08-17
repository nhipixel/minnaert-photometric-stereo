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
    # Accumulated one light at a time. Forming the full (H, W, m) log arrays
    # would allocate hundreds of megabytes of temporaries per object for no
    # benefit, since only these five sums are needed.
    shape = images.shape[:2]
    n = np.zeros(shape, dtype=np.int32)
    sx = np.zeros(shape)
    sy = np.zeros(shape)
    sxx = np.zeros(shape)
    sxy = np.zeros(shape)

    for j in range(lights.shape[0]):
        cos_j = normals @ lights[j]
        ok = (cos_j > min_cos) & (images[..., j] > EPS)
        if not ok.any():
            continue
        x = np.where(ok, np.log(np.clip(cos_j, EPS, None)), 0.0)
        y = np.where(ok, np.log(np.clip(images[..., j], EPS, None)), 0.0)
        n += ok
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y

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
