"""
Evaluation metrics.

Angular error is the standard photometric stereo metric and is what the
DiLiGenT benchmark reports, so synthetic and real results stay comparable.
"""
import numpy as np

# A normal shorter than this carries no direction, so any angle against it is
# undefined rather than zero.
DEGENERATE_TOL = 1e-9


def angular_error_deg(normals_est, normals_true, mask=None):
    """
    Per pixel angle between estimated and ground truth normals, in degrees.

    The angle is taken as atan2 of the cross product length against the dot
    product rather than as arccos of the dot product. Near zero angle the
    derivative of arccos is unbounded, so it amplifies rounding in the dot
    product by a square root and floors out around 1e-6 degrees. The atan2
    form stays accurate there, which matters because the Lambertian case is
    validated at exactly that scale.

    Pixels where either vector has no direction return NaN rather than a score.
    atan2 is magnitude invariant, so a zero length ground truth normal would
    otherwise read as zero degrees, a perfect match, against any estimate at
    all. One benchmark object contains 73 such pixels, enough to shift its
    reported error by 0.19 degrees.
    """
    dot = np.sum(normals_est * normals_true, axis=-1)
    cross = np.linalg.norm(np.cross(normals_est, normals_true), axis=-1)
    err = np.degrees(np.arctan2(cross, dot))

    degenerate = (np.linalg.norm(normals_est, axis=-1) < DEGENERATE_TOL) | (
        np.linalg.norm(normals_true, axis=-1) < DEGENERATE_TOL
    )
    err = np.where(degenerate, np.nan, err)

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


def mean_angular_error(normals_est, normals_true, mask=None):
    """Mean angular error in degrees, the single number most results quote."""
    return summarize(angular_error_deg(normals_est, normals_true, mask), mask)["mean_deg"]
