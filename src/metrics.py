"""
Evaluation metrics.

Angular error is the standard photometric stereo metric and is what the
DiLiGenT benchmark reports, so synthetic and real results stay comparable.
"""
import numpy as np


def angular_error_deg(normals_est, normals_true, mask=None):
    """
    Per pixel angle between estimated and ground truth normals, in degrees.

    The angle is taken as atan2 of the cross product length against the dot
    product rather than as arccos of the dot product. Near zero angle the
    derivative of arccos is unbounded, so it amplifies rounding in the dot
    product by a square root and floors out around 1e-6 degrees. The atan2
    form stays accurate there, which matters because the Lambertian case is
    validated at exactly that scale.
    """
    dot = np.sum(normals_est * normals_true, axis=-1)
    cross = np.linalg.norm(np.cross(normals_est, normals_true), axis=-1)
    err = np.degrees(np.arctan2(cross, dot))
    if mask is not None:
        err = np.where(mask, err, np.nan)
    return err


def summarize(err, mask=None):
    """Mean, median and tail percentiles of an angular error map."""
    vals = err[mask] if mask is not None else err
    vals = vals[np.isfinite(vals)]
    return {
        "mean_deg": float(np.mean(vals)),
        "median_deg": float(np.median(vals)),
        "p90_deg": float(np.percentile(vals, 90)),
        "max_deg": float(np.max(vals)),
        "n_pixels": int(vals.size),
    }


def albedo_relative_error(albedo_est, albedo_true, mask=None):
    """Signed relative albedo error, used to expose the n_z bias."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (albedo_est - albedo_true) / albedo_true
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        rel = np.where(mask, rel, np.nan)
    return rel
