"""hud_tilt against the three dial screenshots in test_shots/tilt (1:1 crops of the
2560x1440 frame, 2026-09-29). Run: python -m pytest

The expected values were measured separately from the reader, by connected
components on colour masks and a per-row fit of the barrel:
  - pitch: the triangles' centre row against the centre tick, over the 22.33 px tick pitch,
    less the 3 px a LEVEL hull shows (PITCH_ZERO_PX, measured 2026-09-30)
  - roll: the angle of the line through the two bar centroids (+ = right end higher)
  - turret: the barrel above the hull, fitted row by row (+ = clockwise, to the right)
"""
import os

import cv2
import numpy as np
import pytest

import hud_tilt as t

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_shots", "tilt")


def img(name):
    return cv2.imread(os.path.join(HERE, name))


@pytest.mark.parametrize("name,pitch,roll,turret,stabilized", [
    # centre tick 158, triangles 160.5; bars (213.6, 159.2)-(347.4, 155.8); barrel 6.9 deg right
    ("stabilized_gunner_asl22.png", -0.24, 1.46, 6.9, True),
    # the USER's nose-up shot: centre tick 157, triangles 122.5; bars (197.1, 165.9)-(329.9, 147.1)
    ("uphill_driving_8kmh.png", 1.42, 8.06, 0.0, False),
    # side-on: centre tick 124, triangles 146.5; bars (195.1, 112.3)-(327.9, 134.7)
    ("sideon_stopped.png", -1.14, -9.57, 0.0, False),
    # LIVE BOX grab 2026-09-29 20:01 that read "no tilt dial": the right scale's second
    # tick touched a bright scene edge and merged into a taller blob, leaving 6 of 7.
    # Centre tick 140.5, triangles 138.5; bars (71.5, 140.5)-(207.0, 140.5); barrel 7.6 right
    ("live_box_tick_merged.png", -0.04, 0.0, 7.6, True),
])
def test_reads_the_dial(name, pitch, roll, turret, stabilized):
    got, why = t.read(img(name))
    assert why == "" and got is not None
    assert got.pitch_ticks == pytest.approx(pitch, abs=0.06)
    assert got.roll_deg == pytest.approx(roll, abs=0.5)
    assert got.turret_deg == pytest.approx(turret, abs=1.5)
    assert got.stabilized is stabilized


@pytest.mark.parametrize("name,turret", [
    # 2026-10-14 multiplayer turret sweep, the turret's truth = tape - minimap hull heading - 0.97:
    # pointing back, the barrel runs beside the turret's outline (read 144.5 before "either side")
    ("live_turret_back_152.png", 151.9),
    # pale concrete: no dial found at INK, read at INK_PALE
    ("live_pale_concrete_turret_0.png", 0.3),
])
def test_reads_the_turret_on_recorded_dials(name, turret):
    got, why = t.read(img(name))
    assert why == "" and got is not None
    assert got.turret_deg == pytest.approx(turret, abs=2.0)


def test_pitch_in_degrees_uses_the_tick_scale():
    got, _ = t.read(img("uphill_driving_8kmh.png"))
    assert got.pitch_deg == pytest.approx(got.pitch_ticks * t.DEG_PER_TICK)


def test_reads_it_off_a_whole_frame_through_the_fixed_box():
    # The stabilized shot is a 1:1 crop whose top-left sits at frame (1000, 1103): its
    # ASL digits end at crop x 129 / start at row 148, the frame's at 1129 / 1251.
    crop = img("stabilized_gunner_asl22.png")
    frame = np.zeros((1440, 2560, 3), np.uint8)
    frame[1103:1103 + crop.shape[0], 1000:1000 + crop.shape[1]] = crop
    x, y, w, h = t.BOX
    got, why = t.read(frame[y:y + h, x:x + w])
    assert why == "" and got.turret_deg == pytest.approx(6.9, abs=1.5)
    assert got.roll_deg == pytest.approx(1.46, abs=0.5)


def test_no_dial_is_a_reason_not_a_reading():
    got, why = t.read(np.full((250, 280, 3), 40, np.uint8))
    assert got is None and "dial" in why


def test_a_dial_with_its_ticks_cut_off_is_refused():
    # Only the top half of the scales: the centre tick cannot be told from the others.
    got, why = t.read(img("sideon_stopped.png")[:118])
    assert got is None and why


def test_a_level_hull_reads_level():
    # 2026-09-30: the hull pivoted a full turn on flat concrete read +0.134 ticks at every
    # heading before the zero; the dial said "Unstabilized" (not asserted: see the docstring)
    got, why = t.read(img("flat_concrete_hull_pivot.png"))
    assert why == ""
    assert got.pitch_ticks == pytest.approx(0.0, abs=0.02)
    assert got.roll_deg == pytest.approx(0.0, abs=0.3)
