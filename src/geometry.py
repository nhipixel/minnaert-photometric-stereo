"""
Synthetic surface geometry with analytic ground truth normals.

A unit sphere is used because its normals are known in closed form, so the
ground truth carries no discretization error of its own.
"""
import numpy as np


def sphere_normals(resolution=256, radius_max=0.98):
    """
    Unit sphere viewed orthographically down the z axis.

    For a point (x, y) inside the unit disc the outward normal of the unit
    sphere is exactly (x, y, sqrt(1 - x^2 - y^2)).

    radius_max trims a thin rim off the silhouette. At the exact silhouette
    n_z is zero and the Minnaert prefactor n_z^(c-1) is unbounded for c < 1,
    which would produce infinities rather than a measurable trend.

    Returns
        normals  float64 array of shape (resolution, resolution, 3)
        mask     bool array, True where the pixel lies on the sphere
    """
    lin = np.linspace(-1.0, 1.0, resolution)
    x, y = np.meshgrid(lin, lin)
    r2 = x * x + y * y
    mask = r2 <= radius_max * radius_max

    z = np.zeros_like(x)
    z[mask] = np.sqrt(1.0 - r2[mask])

    normals = np.stack([x, y, z], axis=-1)
    normals[~mask] = 0.0
    return normals, mask


def checker_albedo(resolution=256, squares=8, low=0.4, high=0.9):
    """
    Spatially varying albedo field.

    Used to confirm that the solver separates albedo from orientation. A
    uniform albedo would hide a bug that couples the two.
    """
    lin = np.arange(resolution)
    gx, gy = np.meshgrid(lin, lin)
    cell = resolution // squares
    checker = ((gx // cell) + (gy // cell)) % 2
    return np.where(checker == 0, low, high).astype(np.float64)


def flatten_masked(field, mask):
    """Collapse a (H, W, ...) field to (N, ...) keeping only masked pixels."""
    return field[mask]
