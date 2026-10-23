"""tilt_fit: the plane fit recovers a known slope. Run: python -m pytest"""
import math

import pytest

import tilt_fit


def _plane(uphill, g_ticks, g_roll, p0=0.13, r0=0.0, n=360, noise=0.0):
    import random
    rnd = random.Random(1)
    rows = []
    for i in range(n):
        h = i * 360 / n
        d = math.radians(h - uphill)
        rows.append((h, p0 + g_ticks * math.cos(d) + rnd.gauss(0, noise),
                     r0 + g_roll * math.sin(d) + rnd.gauss(0, noise)))
    return rows


def test_fit_recovers_the_uphill_heading_and_both_amplitudes():
    f = tilt_fit.fit(_plane(14.0, 0.45, 3.0, noise=0.02))
    assert f["uphill_heading"] == pytest.approx(14.0, abs=1.0)
    assert f["pitch_amp_ticks"] == pytest.approx(0.45, abs=0.02)
    assert f["roll_amp_deg"] == pytest.approx(3.0, abs=0.1)
    assert f["pitch_zero_ticks"] == pytest.approx(0.13, abs=0.02)
    assert f["roll_phase_vs_pitch_deg"] == pytest.approx(90, abs=3)
    assert f["deg_per_tick_if_roll_true"] == pytest.approx(3.0 / 0.45, rel=0.05)


def test_a_reversed_roll_shows_as_minus_90():
    f = tilt_fit.fit(_plane(200.0, 0.4, -2.5))
    assert f["roll_phase_vs_pitch_deg"] == pytest.approx(-90, abs=3)


def test_a_partial_turn_is_refused():
    with pytest.raises(ValueError):
        tilt_fit.fit([r for r in _plane(0, 0.4, 3) if r[0] < 120])


def test_bearing_unwraps_past_north():
    assert tilt_fit.unwrap_bearing(358.0, 2.0) == 362.0
    assert tilt_fit.unwrap_bearing(2.0, 358.0) == -2.0
