import glob
import os
import re

import cv2
import numpy as np
import pytest

import hud_bearing

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = sorted(glob.glob(os.path.join(HERE, "test_shots", "bearing_check", "b*.png")))


def _want(p):
    return float(re.match(r"b(\d+\.\d+)", os.path.basename(p)).group(1))


def _off(a, b):
    return abs((a - b + 180) % 360 - 180)


@pytest.mark.parametrize("path", CHECK, ids=os.path.basename)
@pytest.mark.parametrize("miss", [-6.0, 0.0, 6.0])
def test_held_out_traverse_frames_read_from_a_near_bearing(path, miss):
    # 30 frames of the 2026-09-30 traverses not used for the labels; `near` up to 6 deg off.
    # Where the labels are clear enough to disagree they may refuse, but never misread.
    b, why = hud_bearing.read(cv2.imread(path), (_want(path) + miss) % 360)
    if b is None:
        assert "labels say" in why
    else:
        assert _off(b.deg, _want(path)) < 0.01


def test_most_held_out_frames_read():
    got = sum(hud_bearing.read(cv2.imread(p), _want(p))[0] is not None for p in CHECK)
    assert len(CHECK) >= 20 and got >= len(CHECK) - 2


def test_no_near_bearing_is_a_reason():
    b, why = hud_bearing.read(cv2.imread(CHECK[0]), None)
    assert b is None and "near" in why


def test_a_frame_without_the_tape_is_a_reason_not_a_reading():
    blank = np.full((hud_bearing.BOX[3], hud_bearing.BOX[2], 3), 90, np.uint8)
    b, why = hud_bearing.read(blank, 100.0)
    assert b is None and why


def test_clear_labels_veto_a_step_fifteen_degrees_off():
    caught = [p for p in CHECK if hud_bearing.read(cv2.imread(p), _want(p) + 15)[0] is None]
    assert caught                     # some frames' labels are clear enough to say so
    b, why = hud_bearing.read(cv2.imread(caught[0]), _want(caught[0]) + 15)
    assert "compass" in why


def test_build_reproduces_the_committed_labels(tmp_path):
    out = tmp_path / "l.npz"
    counts, used = hud_bearing.build(out=str(out))
    assert used >= 80 and len(counts) == 24
    with np.load(out) as new, np.load(hud_bearing.LABELS) as old:
        assert sorted(new.files) == sorted(old.files)
        for k in new.files:
            assert np.allclose(new[k], old[k])


def test_observe_gives_the_position_in_the_step_exactly():
    for p in CHECK:
        obs, why = hud_bearing.observe(cv2.imread(p))
        assert obs is not None and why == ""
        assert -7.5 <= obs.frac <= 7.5
        truth = _want(p)
        k = round((truth - obs.frac) / 15)
        assert _off((15 * k + obs.frac) % 360, truth) < 0.01   # some step k is exactly right
        assert len(obs.candidates()) == 24 and len(obs.labels) == 24


def test_observe_labels_point_at_the_true_step_more_often_than_not():
    hits = 0
    for p in CHECK:
        obs, _ = hud_bearing.observe(cv2.imread(p))
        k_true = round((_want(p) - obs.frac) / 15) % 24
        hits += int(np.argmax(obs.labels)) == k_true
    assert hits >= len(CHECK) // 2       # weak evidence alone (57% held out, 2026-10-11)


def test_evidence_adds_compass_and_weighted_labels():
    obs = hud_bearing.TapeObs(0.0, [1.0] + [0.0] * 23, [0.0, 2.0] + [0.0] * 22)
    ev = obs.evidence()
    assert ev[0] == pytest.approx(hud_bearing.LABEL_WEIGHT) and ev[1] == pytest.approx(2.0)
    assert hud_bearing.TapeObs(1.0, None, None).evidence() is None


def test_observe_without_the_tape_is_a_reason():
    blank = np.full((hud_bearing.BOX[3], hud_bearing.BOX[2], 3), 90, np.uint8)
    obs, why = hud_bearing.observe(blank)
    assert obs is None and why


def test_the_recorder_writes_numpy_numbers(tmp_path):
    # 2026-10-12 live: the minimap recorder logged a float32 icon heading and every tick raised
    import json
    import numpy as np
    r = hud_bearing.Recorder(str(tmp_path))
    r.save({}, {"heading": np.float32(12.5), "n": np.int64(3), "xy": np.array([1.0, 2.0])})
    line = json.loads((tmp_path / "log.jsonl").read_text())
    assert line["heading"] == 12.5 and line["n"] == 3 and line["xy"] == [1.0, 2.0]


def test_a_trusted_prior_replaces_an_unconfirmed_untrusted_start():
    # review 2026-10-12 I5: after the map, the first tape read came before the tilt dial -- an
    # untrusted start on the target's azimuth that a gun at rest never confirms
    tr = hud_bearing.StepTracker()
    ev = [1.0] * 24
    assert tr.update(0.0, 5.0, ev, 100.0, trusted=False).deg is None
    t = 0.0
    for _ in range(hud_bearing.SETTLE_READS + 1):
        t += 0.03
        r = tr.update(t, 5.0, ev, 125.0, trusted=True)
    assert r.deg == 125.0



# --- the black labels over a bright sky prove the step (USER 2026-10-12) ------------------------
PROOF = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_shots",
                                      "bearing_proof", "b*.png")))


def _proof_true_k(path, obs):
    deg = float(re.match(r"b(\d+\.\d+)", os.path.basename(path)).group(1))
    return round((deg - obs.frac) / 15) % 24


def test_clear_black_labels_on_a_bright_sky_prove_the_right_step():
    bright = [p for p in PROOF if "_bright_" in p]
    assert len(bright) >= 5
    for path in bright:
        obs, why = hud_bearing.observe(cv2.imread(path))
        assert obs is not None, why
        assert obs.proof == _proof_true_k(path, obs), path


def test_labels_over_dark_ground_never_prove_a_wrong_step():
    # and the traps: frames whose labels picked a wrong step confidently on held-out sessions
    # (2026-10-12: 0.126 on a frame showing 270 and 285 plainly -- the cut was 0.12)
    dark = [p for p in PROOF if "_dark_" in p or "_trap_" in p]
    assert dark
    for path in dark:
        obs, why = hud_bearing.observe(cv2.imread(path))
        assert obs is None or obs.proof in (None, _proof_true_k(path, obs)), path


def test_two_proof_reads_settle_an_untrusted_start():
    tr = hud_bearing.StepTracker()
    ev = [1.0] * 24
    assert tr.update(0.0, 5.0, ev, 100.0, trusted=False, proof=8).deg is None     # one is not enough
    r = tr.update(0.03, 5.0, ev, 100.0, trusted=False, proof=8)
    assert r.deg == 125.0 and tr.label_proven


def test_proof_reads_must_come_in_a_row():
    tr = hud_bearing.StepTracker()
    ev = [1.0] * 24
    tr.update(0.0, 5.0, ev, 100.0, proof=8)
    tr.update(0.03, 5.0, ev, 100.0, proof=None)
    r = tr.update(0.06, 5.0, ev, 100.0, proof=8)
    assert r.deg is None and not tr.label_proven


def test_two_proof_reads_put_a_wrong_track_right():
    # a trusted start on a wrong step (the flipped icon of 2026-10-12) -- the labels correct it
    tr = hud_bearing.StepTracker()
    ev = [1.0] * 24
    t = 0.0
    for _ in range(hud_bearing.SETTLE_READS + 1):
        r = tr.update(t, 5.0, ev, 110.0, trusted=True)
        t += 0.03
    assert r.deg == 110.0
    tr.update(t, 5.0, ev, 110.0, trusted=True, proof=8)
    r = tr.update(t + 0.03, 5.0, ev, 110.0, trusted=True, proof=8)
    assert r.deg == 125.0 and tr.label_proven



def test_the_learned_label_examples_are_used():
    assert hud_bearing._label_nn() is not None


def test_a_direction_with_few_learned_examples_proves_nothing(monkeypatch):
    # bearing_labels_nn.py: today's sessions barely faced 15-60 and 165-180 -- the zero-wrong cut
    # was measured where there were examples, so elsewhere the labels vote but never prove
    path = next(p for p in PROOF if "_bright_" in p)
    X, Y = hud_bearing._label_nn()
    obs, _ = hud_bearing.observe(cv2.imread(path))
    assert obs.proof is not None
    k = obs.proof
    v = (15 * k) % 360                                   # the label at the sight's long tick
    keep = (Y != v) | (np.cumsum(Y == v) <= 3)           # that value: 3 examples left
    monkeypatch.setattr(hud_bearing, "_label_nn_store", (X[keep], Y[keep]))
    obs2, _ = hud_bearing.observe(cv2.imread(path))
    assert obs2.proof is None
