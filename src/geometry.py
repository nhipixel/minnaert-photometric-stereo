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


def perspective_view_field(resolution=256, radius_max=0.98, camera_distance=4.0):
    """
    Unit vector from each sphere point toward a finite camera on the z axis.

    Under orthographic viewing every pixel shares the view direction z, so
    cos(theta_r) equals n_z everywhere. With a finite camera the view direction
    varies across the image, which is the general case the reduction is claimed
    to cover: what it needs is that the viewpoint is fixed across the light
    stack, not that projection is orthographic.
    """
    lin = np.linspace(-1.0, 1.0, resolution)
    x, y = np.meshgrid(lin, lin)
    r2 = x * x + y * y
    mask = r2 <= radius_max * radius_max

    z = np.zeros_like(x)
    z[mask] = np.sqrt(1.0 - r2[mask])

    view = np.stack([-x, -y, camera_distance - z], axis=-1)
    view /= np.linalg.norm(view, axis=-1, keepdims=True)
    return view


def bump_normals(resolution=256, n_bumps=7, amplitude=0.09, sigma=0.16, seed=0):
    """
    Smooth height field of summed Gaussians, viewed orthographically.

    A second geometry is needed because every result measured on a sphere is
    open to the objection that it is a property of that shape. Here the height
    is z = sum_k A exp(-r_k^2 / 2s^2), so the surface gradients, and therefore
    the normals, are still available in closed form:

        n proportional to (-dz/dx, -dz/dy, 1)

    Amplitude is kept small so the surface stays a graph with n_z bounded well
    away from zero, which keeps the Minnaert prefactor finite everywhere.

    Returns normals of shape (resolution, resolution, 3) and an all True mask,
    since every pixel of a height field is valid.
    """
    lin = np.linspace(-1.0, 1.0, resolution)
    x, y = np.meshgrid(lin, lin)

    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.75, 0.75, size=(n_bumps, 2))

    dzdx = np.zeros_like(x)
    dzdy = np.zeros_like(y)
    for cx, cy in centers:
        dx, dy = x - cx, y - cy
        gauss = amplitude * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        dzdx += -dx / (sigma * sigma) * gauss
        dzdy += -dy / (sigma * sigma) * gauss

    normals = np.stack([-dzdx, -dzdy, np.ones_like(x)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals, np.ones(x.shape, dtype=bool)


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
