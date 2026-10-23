"""grid_align: a known map-vs-grid offset is recovered from tilt readings. Run: python -m pytest"""
import math
import os

import pytest

import grid_align
import terrain

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(terrain.DIR, "bakurani", "manifest.json")),
                                reason="terrain data not present")


def _reading(grid, x, y, heading):
    dzdx, dzdy = grid_align.slopes(grid, x, y, half_m=16)
    c = dzdx.shape[0] // 2
    p, r = grid_align.predicted(dzdx, dzdy, heading)
    return float(p[c, c]), float(r[c, c])


def test_predicted_signs_match_the_dial():
    import numpy as np
    east_slope = (np.array([[0.2]]), np.array([[0.0]]))       # ground rises to the east
    p, r = grid_align.predicted(*east_slope, 90)
    assert p[0, 0] == pytest.approx(math.degrees(math.atan(0.2))) and r[0, 0] == pytest.approx(0, abs=1e-9)
    p, r = grid_align.predicted(*east_slope, 180)             # facing south: east is on the LEFT
    assert p[0, 0] == pytest.approx(0, abs=1e-9) and r[0, 0] < 0
    p, r = grid_align.predicted(*east_slope, 0)               # facing north: east is on the RIGHT
    assert r[0, 0] > 0


def test_fit_recovers_a_known_offset_on_varied_ground():
    grid = terrain.load("bakurani")
    true_offset = (24.0, -16.0)                                # metres, map -> grid
    spots = [(85.66, 32.98, 203), (85.0, 32.4, 20), (86.6, 33.2, 110), (84.9, 32.8, 300),
             (83.63, 19.29, 162), (99.0, 108.5, 14), (99.3, 108.9, 250)]
    stations = []
    for x, y, h in spots:
        gx, gy = x + true_offset[0] / 100, y + true_offset[1] / 100          # where the tank truly is
        p, r = _reading(grid, gx, gy, h)
        stations.append({"x": x, "y": y, "heading": h, "pitch_deg": p, "roll_deg": r, "map": "bakurani"})
    res = grid_align.fit(stations)
    assert res["east"] == pytest.approx(true_offset[0], abs=4)
    assert res["north"] == pytest.approx(true_offset[1], abs=4)
    assert res["east_range"][0] <= true_offset[0] <= res["east_range"][1]


def test_fit_refuses_stations_from_two_maps():
    a = {"x": 1, "y": 1, "heading": 0, "pitch_deg": 0, "roll_deg": 0, "map": "bakurani"}
    with pytest.raises(ValueError):
        grid_align.fit([a, dict(a, map="ozeti")])
