"""
Gate for the reproduced landscape of published estimate maps.

The table is only usable if every cell is real and every row is scored on the
same pixels against the same ground truth. A missing map or a divergent
convention would otherwise surface as a plausible looking but wrong ranking.
"""
import numpy as np
import pytest

from diligent import (
    OBJECTS,
    PUBLISHED_BASELINE,
    PUBLISHED_METHODS,
    is_available,
    load_ground_truth,
    load_published_estimate,
)
from metrics import mean_angular_error

pytestmark = pytest.mark.skipif(
    not is_available(), reason="DiLiGenT not downloaded, see README"
)


@pytest.fixture(scope="module")
def truth():
    return {name: load_ground_truth(name) for name in OBJECTS}


@pytest.fixture(scope="module")
def table(truth):
    out = {}
    for method in PUBLISHED_METHODS:
        for name in OBJECTS:
            gt, mask = truth[name]
            out[(method, name)] = mean_angular_error(
                load_published_estimate(name, method), gt, mask
            )
    return out


# Averages from the literature for the eight rescored methods, used to pin the
# 80 non baseline cells to something better than a plausibility band. Small
# deviations are expected since degenerate ground truth pixels are excluded
# here and were not in the original evaluations.
PUBLISHED_AVERAGES = {
    "CVPR12Shi": 10.30, "CVPR14Ikehata": 10.60, "ICCV05Goldman": 10.77,
    "CVPR10Higo": 12.42, "CVPR08Alldrin": 12.59, "ACCV10Wu": 13.33,
    "CVPR12Ikehata": 13.66, "ECCV12Shi": 14.57,
}


def test_every_cell_exists_and_is_plausible(table):
    for key, value in table.items():
        assert np.isfinite(value), key
        assert 0.5 < value < 60.0, (key, value)


@pytest.mark.parametrize("method", sorted(PUBLISHED_AVERAGES))
def test_each_rescored_method_lands_on_its_published_average(table, method):
    """
    Pins the 80 cells that the baseline row does not cover. Without this the
    only value level check in the table is the l2 row, and the rest could drift
    without any test noticing.
    """
    avg = float(np.mean([table[(method, n)] for n in OBJECTS]))
    assert abs(avg - PUBLISHED_AVERAGES[method]) < 0.05, (method, avg)


def test_l2_row_matches_the_tabulated_baseline(table):
    """
    The shipped l2 maps rescored here must land on the tabulated numbers,
    which ties the whole table to the same convention as the rest of the
    report. pot2 differs by its documented degenerate pixel exclusion.
    """
    for name in OBJECTS:
        tolerance = 0.25 if name == "pot2" else 0.02
        assert abs(table[("l2", name)] - PUBLISHED_BASELINE[name]) < tolerance, name


def test_every_published_method_beats_the_baseline_on_average(table):
    l2_avg = np.mean([table[("l2", n)] for n in OBJECTS])
    for method in PUBLISHED_METHODS:
        if method == "l2":
            continue
        avg = np.mean([table[(method, n)] for n in OBJECTS])
        assert avg < l2_avg, (method, avg, l2_avg)


def test_the_spread_of_a_decade_is_modest(table):
    """
    The distance between the best published method and the plain baseline is
    what any correction has to be read against, so its size is pinned here
    rather than quoted loosely.
    """
    averages = {m: np.mean([table[(m, n)] for n in OBJECTS]) for m in PUBLISHED_METHODS}
    best = min(averages.values())
    gain = averages["l2"] - best
    assert 4.0 < gain < 7.0, gain
