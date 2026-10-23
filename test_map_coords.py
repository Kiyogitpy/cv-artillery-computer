"""map_coords against the labelled map captures in test_shots/map, named
x<x>_y<y>_at<crossing x>-<crossing y>. Run: python -m pytest"""
import os

import cv2
import numpy as np
import pytest

import map_coords as m

SAMPLES = m.samples()
IDS = [os.path.basename(p) for p, _, _ in SAMPLES]


def truth(t):
    return float(t["x"]), float(t["y"])


def test_there_are_samples_and_every_digit_is_covered():
    assert len(SAMPLES) >= 6
    assert set(m.Glyphs().labels) == set("0123456789")


@pytest.mark.parametrize("path,t,at", SAMPLES, ids=IDS)
def test_reads_capture(path, t, at):
    r = m.read(cv2.imread(path), at)
    assert (r.x, r.y) == truth(t) and r.why == ""


@pytest.mark.parametrize("path,t,at", SAMPLES, ids=IDS)
def test_reads_capture_it_was_not_built_from(path, t, at, tmp_path):
    out = tmp_path / "glyphs.npz"
    m.build(out=str(out), skip=(os.path.basename(path),))
    r = m.read(cv2.imread(path), at, glyphs=m.Glyphs(str(out)))
    assert (r.x, r.y) == truth(t)


@pytest.mark.parametrize("path,t,at", SAMPLES, ids=IDS)
def test_a_cursor_a_few_px_off_the_crosshair_still_reads(path, t, at):
    im = cv2.imread(path)
    for dx, dy in ((-6, 0), (6, 0), (0, -6), (0, 6), (4, -4), (-5, 5)):
        r = m.read(im, (at[0] + dx, at[1] + dy))
        assert (r.x, r.y) == truth(t), (dx, dy, r)


@pytest.mark.parametrize("path,t,at", SAMPLES, ids=IDS)
def test_away_from_the_labels_nothing_is_invented(path, t, at):
    im = cv2.imread(path)
    for dx, dy in ((200, 0), (-150, 120), (0, 250), (90, 60), (-60, -40)):
        r = m.read(im, (at[0] + dx, at[1] + dy))
        assert r.x is None and r.y is None and r.why, (dx, dy, r)


@pytest.mark.parametrize("path,t,at", SAMPLES, ids=IDS)
def test_reads_the_grab_artillery_takes_around_the_cursor(path, t, at):
    dx, dy, w, h = m.GRAB
    left, top = at[0] + dx, at[1] + dy
    crop = cv2.imread(path)[top:top + h, left:left + w]
    r = m.read(crop, at, origin=(left, top))
    assert (r.x, r.y) == truth(t)


def test_committed_glyphs_match_a_fresh_build(tmp_path):
    out = tmp_path / "glyphs.npz"
    m.build(out=str(out))
    a, b = np.load(m.GLYPHS), np.load(str(out))
    for k in ("letter_x", "letter_y", "digits"):
        assert np.allclose(a[k], b[k]), k
    assert list(a["labels"]) == list(b["labels"])



RED = m.samples(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_shots", "map_red"))


@pytest.mark.parametrize("path,t,at", RED, ids=[os.path.basename(p) for p, _, _ in RED])
def test_reads_a_label_over_a_red_held_zone(path, t, at):
    # live 2026-10-12: with red holding the zone the map turns pink -- light enough (brightness ~180,
    # its brightest tenth ~211) that 500 background pixels passed the fixed white-text mask
    r = m.read(cv2.imread(path), at)
    assert (r.x, r.y) == (float(t["x"]), float(t["y"])), r.why
