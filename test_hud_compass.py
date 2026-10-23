"""hud_compass: the top compass box settles the bearing tape's 15-degree step. Run: python -m pytest"""
import glob
import os
import re

import cv2
import pytest

import hud_compass as hc

pytestmark = pytest.mark.skipif(not os.path.exists(hc.DIGITS) or not glob.glob(os.path.join(hc.SAMPLES, "c*.png")),
                                reason="compass samples not recorded")


def test_the_box_number_is_the_tape_less_078_rounded_down():
    assert hc.expected_box(353.78) == 353
    assert hc.expected_box(0.5) == 359            # the box reads below the tape, across north too
    assert hc.expected_box(1.0) == 0
    assert hc.expected_box(300.78) == 300 and hc.expected_box(300.77) == 299


def _frames(n=60):
    files = sorted(glob.glob(os.path.join(hc.SAMPLES, "c*.png")))[::max(1, len(glob.glob(os.path.join(hc.SAMPLES, "c*.png"))) // n)]
    for p in files:
        yield int(re.match(r"c(\d{3})_", os.path.basename(p)).group(1)), cv2.imread(p)


def test_choose_picks_the_right_step_among_24_candidates():
    right = total = 0
    for lab, img in _frames():
        truth = lab + 1.28                              # the tape reads about this for that box number
        cands = [(truth + 15 * j) % 360 for j in range(-12, 12)]
        best, margin, why = hc.choose(img, cands)
        total += 1
        right += best is not None and abs((best - truth + 180) % 360 - 180) < 1.0
    assert right / total >= 0.9                          # in-sample; the held-out rate was 93%, 97% at margin 0.1


def test_a_blank_box_is_a_reason_not_a_reading():
    import numpy as np
    blank = np.zeros((hc.BOX[3], hc.BOX[2], 3), np.uint8)
    deg, why = hc.read(blank)
    assert deg is None and why


def test_totals_score_every_candidate_and_choose_agrees():
    for lab, img in list(_frames(10)):
        truth = lab + 1.28
        cands = [(truth + 15 * j) % 360 for j in range(-12, 12)]
        tot, why = hc.totals(img, cands)
        assert why == "" and len(tot) == 24
        best, margin, _ = hc.choose(img, cands)
        assert best == cands[max(range(24), key=lambda i: tot[i])]


def test_totals_of_a_blank_box_is_a_reason():
    import numpy as np
    tot, why = hc.totals(np.zeros((hc.BOX[3], hc.BOX[2], 3), np.uint8), [1.0, 16.0])
    assert tot is None and why
