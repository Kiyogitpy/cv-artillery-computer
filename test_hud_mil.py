import glob
import os
import re

import cv2
import numpy as np
import pytest

import hud_mil

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = sorted(glob.glob(os.path.join(HERE, "test_shots", "mil_check", "mil*.png")))


def _want(p):
    return float(re.match(r"mil(\d+\.\d+)", os.path.basename(p)).group(1))


@pytest.mark.parametrize("path", CHECK, ids=os.path.basename)
def test_held_out_sweep_frames_read_to_the_hundredth(path):
    # 30 frames of the 2026-09-30 sweep NOT used to learn the digits, and the first
    # screenshot (57.13: 50 mils 125.5 px above the centre line, 176 px to 10 mils)
    r, why = hud_mil.read(cv2.imread(path))
    assert r is not None, why
    assert r.mil == pytest.approx(_want(path), abs=0.01)


def test_there_are_held_out_frames():
    assert len(CHECK) >= 20


def test_a_frame_without_the_scale_is_a_reason_not_a_reading():
    blank = np.full((hud_mil.BOX[3], hud_mil.BOX[2], 3), 90, np.uint8)
    r, why = hud_mil.read(blank)
    assert r is None and why


def test_labels_that_disagree_give_no_reading():
    # of two readable labels, one read as a value that is not the other's +-10 per tick:
    # a tie, so no reading (with three, two outvote one)
    class Liar(hud_mil.Templates):
        lied = False

        def number(self, gs):
            v = super().number(gs)
            if v is not None and not Liar.lied:
                Liar.lied = True
                return 5000
            return v

    path = next(p for p in CHECK if hud_mil.read(cv2.imread(p))[0].labels == 2)
    r, why = hud_mil.read(cv2.imread(path), Liar())
    assert r is None and "disagree" in why


def test_build_reproduces_the_committed_templates(tmp_path):
    out = tmp_path / "g.npz"
    counts, used = hud_mil.build(out=str(out))
    assert used >= 40 and all(counts.get(d, 0) >= 2 for d in "0123456789,")
    with np.load(out) as new, np.load(hud_mil.GLYPHS) as old:
        assert sorted(new.files) == sorted(old.files) == ["X", "Y"]     # every example, and its character
        assert (new["Y"] == old["Y"]).all()
        assert np.allclose(new["X"], old["X"])
