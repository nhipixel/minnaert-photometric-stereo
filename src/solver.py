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


def woodham_weighted(images, lights, weights, mask=None):
    """
    Per pixel weighted least squares through the 3x3 normal equations.

    Solving A g = b with A = sum_j w_j s_j s_j^T avoids one pseudoinverse per
    distinct weight pattern, which is what makes per pixel weights tractable
    with 96 lights. A tiny ridge keeps pixels with too few surviving lights
    invertible instead of raising.
    """
    h, w, m = images.shape
    if lights.shape[0] != m:
        raise ValueError(f"images have {m} lights but L has {lights.shape[0]} rows")

    W = weights.reshape(-1, m)
    I = images.reshape(-1, m)
    A = np.einsum("nj,ja,jb->nab", W, lights, lights)
    b = np.einsum("nj,nj,ja->na", W, I, lights)
    A += np.eye(3) * 1e-9

    g = np.linalg.solve(A, b[..., None])[..., 0].reshape(h, w, 3)
    albedo = np.linalg.norm(g, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        normals = g / albedo[..., None]
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)

    if mask is not None:
        normals[~mask] = 0.0
        albedo = np.where(mask, albedo, 0.0)
    return normals, albedo


def woodham_trimmed(images, lights, mask=None, drop_low=0.25, drop_high=0.25):
    """
    Least squares after dropping the darkest and brightest observations at
    each pixel. A standard robust control: the dark tail holds shadows and the
    bright tail holds specular highlights, and neither obeys the diffuse model.
    The trim fractions are conventional, not tuned.
    """
    h, w, m = images.shape
    flat = images.reshape(-1, m)
    order = np.argsort(flat, axis=1)
    keep = np.zeros_like(flat)
    lo, hi = int(m * drop_low), int(m * drop_high)
    np.put_along_axis(keep, order[:, lo:m - hi], 1.0, axis=1)
    return woodham_weighted(images, lights, keep.reshape(h, w, m), mask)


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
