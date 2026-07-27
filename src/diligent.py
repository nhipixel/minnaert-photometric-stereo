"""
Loader for the DiLiGenT photometric stereo benchmark.

Ten objects, 96 calibrated directional lights each, 16 bit PNG at 612x512,
orthographic single view. The orthographic setup matters here: the power law
reduction used elsewhere in this project assumes it, so the theory applies to
this data without an extra assumption.

Expected layout, relative to the repo root:

    data/pmsData/ballPNG/
        001.png ... 096.png
        filenames.txt
        light_directions.txt
        light_intensities.txt
        mask.png
        Normal_gt.mat

Download from https://sites.google.com/site/photometricstereodata/single
The data is not redistributed with this repo.
"""
import os
from dataclasses import dataclass

import numpy as np

# Woodham least squares baseline as tabulated in the photometric stereo
# literature, mean angular error in degrees. Used as an external correctness
# check: these are numbers this project did not produce.
#
# One caveat, established by direct comparison. Nine of these reproduce the
# dataset's own shipped baseline result to within 0.005 degrees. The tenth,
# pot2, does not: the shipped pot2PNG_Normal_l2.mat yields 14.46 rather than
# the tabulated 14.65. The shipped artifact is the verifiable one, so it is
# what the tests assert against, and the report states the difference instead
# of quietly adopting either number.
PUBLISHED_BASELINE = {
    "ball": 4.10,
    "bear": 8.39,
    "buddha": 14.92,
    "cat": 8.41,
    "cow": 25.60,
    "goblet": 18.50,
    "harvest": 30.62,
    "pot1": 8.89,
    "pot2": 14.65,
    "reading": 19.80,
}
PUBLISHED_AVERAGE = 15.39

OBJECTS = tuple(PUBLISHED_BASELINE.keys())

# The benchmark's own baseline code collapses colour with rgb2gray, not a plain
# channel mean. The commented out mean in its L2_PMS.m shows the choice was
# deliberate, so these BT.601 luma weights are used to stay comparable with the
# published numbers.
LUMA_BT601 = np.array([0.2989, 0.5870, 0.1140])

# Images are 16 bit and the reference pipeline divides by the full range before
# anything else. This is a global scale that cannot change a normal direction,
# but it keeps intensities in a sane range and matches the reference exactly.
BIT_DEPTH_SCALE = float(2**16 - 1)

# Objects whose reflectance is close enough to diffuse that a single Minnaert
# exponent is expected to describe them. The rest carry strong specular lobes,
# which are a different model class rather than a smaller exponent.
DIFFUSE_OBJECTS = ("ball", "cat", "pot1", "bear")


@dataclass
class DiLiGenTObject:
    name: str
    images: np.ndarray  # (H, W, m) scalar radiance, light intensity divided out
    lights: np.ndarray  # (m, 3) unit directions
    mask: np.ndarray  # (H, W) bool
    normals_gt: np.ndarray  # (H, W, 3) unit vectors, zero outside the mask


def default_root():
    """
    Repo relative location of the extracted benchmark.

    The archive unpacks to data/DiLiGenT/pmsData, which is preferred because it
    keeps the shipped sample code and reference results alongside the data. A
    flattened data/pmsData is accepted as a fallback.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nested = os.path.join(here, "data", "DiLiGenT", "pmsData")
    if os.path.isdir(nested):
        return nested
    return os.path.join(here, "data", "pmsData")


def is_available(root=None):
    return os.path.isdir(root or default_root())


def _read_matrix(path, width):
    """
    Read a whitespace separated numeric file as an (n, width) array.

    The dataset documentation describes these as 3 by 96 matrices but the files
    are written 96 rows of 3. Both orientations are accepted so a transposed
    release does not silently produce wrong light directions.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != width and data.shape[0] == width:
        data = data.T
    if data.shape[1] != width:
        raise ValueError(f"{path} has shape {data.shape}, expected one axis of length {width}")
    return data


def _read_png(path):
    """
    Read a 16 bit PNG as float.

    OpenCV is used rather than Pillow because Pillow does not reliably preserve
    16 bit depth on multichannel PNGs. Channels are returned in RGB order.
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[:, :, ::-1]
    return img.astype(np.float64)


def load_object(name, root=None):
    """
    Load one benchmark object.

    Each observation is divided by that light's measured intensity before the
    colour channels are combined, which is the normalization the benchmark
    expects. Skipping it biases every object.

    The combined grey value is then clipped at 1. This looks arbitrary but is
    required to reproduce the published baseline: the reference implementation
    collapses colour with Matlab rgb2gray, which clamps double output to the
    unit interval. Without the clip, saturated specular pixels disagree with the
    reference by up to 16 degrees and the ball error reads 4.17 instead of 4.10.
    """
    root = root or default_root()
    folder = os.path.join(root, f"{name}PNG")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"object folder not found: {folder}")

    with open(os.path.join(folder, "filenames.txt"), encoding="utf-8") as fh:
        names = [line.strip() for line in fh if line.strip()]

    lights = _read_matrix(os.path.join(folder, "light_directions.txt"), 3)
    intensities = _read_matrix(os.path.join(folder, "light_intensities.txt"), 3)
    if len(names) != lights.shape[0]:
        raise ValueError(f"{name}: {len(names)} images but {lights.shape[0]} light directions")

    lights = lights / np.linalg.norm(lights, axis=1, keepdims=True)

    stack = []
    for fname, intensity in zip(names, intensities):
        img = _read_png(os.path.join(folder, fname)) / BIT_DEPTH_SCALE
        if img.ndim == 3:
            img = np.maximum(img / intensity, 0.0)
            stack.append(np.minimum(img @ LUMA_BT601, 1.0))
        else:
            stack.append(np.clip(img / intensity.mean(), 0.0, 1.0))
    images = np.stack(stack, axis=-1)

    mask = _read_png(os.path.join(folder, "mask.png"))
    if mask.ndim == 3:
        mask = mask @ LUMA_BT601
    # The reference code selects mask == 1 exactly, which drops antialiased
    # edge pixels. Matching that keeps the pixel set identical.
    mask = mask >= mask.max()

    normals_gt = _load_normals(folder)
    normals_gt = np.where(mask[..., None], normals_gt, 0.0)

    return DiLiGenTObject(name, images, lights, mask, normals_gt)


def load_official_l2(name, root=None):
    """
    The dataset's own precomputed Woodham baseline normal map.

    Shipped in estNormalNonLambert alongside results from published methods.
    Comparing against this is stronger than matching a tabulated mean, because
    it checks the estimate pixel by pixel rather than in aggregate.
    """
    from scipy.io import loadmat

    root = root or default_root()
    folder = os.path.join(os.path.dirname(root), "estNormalNonLambert")
    mat = loadmat(os.path.join(folder, f"{name}PNG_Normal_l2.mat"))
    arrays = [v for k, v in mat.items() if not k.startswith("__") and np.ndim(v) == 3]
    if len(arrays) != 1:
        raise KeyError(f"expected one normal array for {name}, got keys {list(mat)}")
    return np.asarray(arrays[0], dtype=np.float64)


def official_l2_available(root=None):
    root = root or default_root()
    return os.path.isdir(os.path.join(os.path.dirname(root), "estNormalNonLambert"))


def _load_normals(folder):
    """Ground truth normals ship as a Matlab matrix under the key Normal_gt."""
    from scipy.io import loadmat

    mat = loadmat(os.path.join(folder, "Normal_gt.mat"))
    for key in ("Normal_gt", "normal_gt", "Normal", "gt_normal"):
        if key in mat:
            return np.asarray(mat[key], dtype=np.float64)
    arrays = [v for k, v in mat.items() if not k.startswith("__") and np.ndim(v) == 3]
    if len(arrays) == 1:
        return np.asarray(arrays[0], dtype=np.float64)
    raise KeyError(f"no ground truth normal array in {folder}, keys were {list(mat)}")
