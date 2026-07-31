"""
Woodham photometric stereo solver.

For non shadowed pixels of a diffuse surface,

    I_k = rho (n dot s_k)

which is linear in g = rho n. Stacking the m light directions into L gives
I = L g, solved by linear least squares. The albedo is the length of g and the
normal is its direction.
"""
import numpy as np


def woodham_lstsq(images, lights, mask=None):
    """
    Solve I = L g for every pixel at once.

    images  (H, W, m) radiance stack
    lights  (m, 3) unit light directions
    mask    optional (H, W) bool, pixels outside it are left at zero

    Returns
        normals  (H, W, 3) unit vectors, zero outside the mask
        albedo   (H, W) the length of g
    """
    h, w, m = images.shape
    if lights.shape[0] != m:
        raise ValueError(f"images have {m} lights but L has {lights.shape[0]} rows")

    # g = pinv(L) I per pixel, written as one matrix product over all pixels.
    flat = images.reshape(-1, m)
    g = flat @ np.linalg.pinv(lights).T
    g = g.reshape(h, w, 3)

    albedo = np.linalg.norm(g, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        normals = g / albedo[..., None]
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)

    if mask is not None:
        normals[~mask] = 0.0
        albedo = np.where(mask, albedo, 0.0)
    return normals, albedo


def woodham_shadow_aware(images, lights, mask=None, threshold=0.0):
    """
    Same solver but each pixel drops the lights whose measured radiance is at
    or below threshold, since a shadowed observation does not satisfy the
    linear model. Falls back to all lights when fewer than three survive.
    """
    h, w, m = images.shape
    if lights.shape[0] != m:
        # Without this the per pixel light subsets still index legally and the
        # solve returns a plausible but wrong answer instead of failing.
        raise ValueError(f"images have {m} lights but L has {lights.shape[0]} rows")

    flat = images.reshape(-1, m)
    g = np.zeros((flat.shape[0], 3))

    valid = flat > threshold
    # Group pixels by which lights they kept so each pattern is solved once.
    patterns, inverse = np.unique(valid, axis=0, return_inverse=True)
    for idx, pattern in enumerate(patterns):
        rows = inverse == idx
        keep = np.flatnonzero(pattern)
        if keep.size < 3:
            keep = np.arange(m)
        sub = np.linalg.pinv(lights[keep])
        g[rows] = flat[np.ix_(rows, keep)] @ sub.T

    g = g.reshape(h, w, 3)
    albedo = np.linalg.norm(g, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        normals = g / albedo[..., None]
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)

    if mask is not None:
        normals[~mask] = 0.0
        albedo = np.where(mask, albedo, 0.0)
    return normals, albedo
