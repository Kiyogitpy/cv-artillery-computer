"""hud_bearing.StepTracker on synthetic reads. Run: python -m pytest"""
import numpy as np

import hud_bearing as hb

DT = 0.25   # the artillery window's bearing poll


def split(deg):
    """(k, frac) the tape gives for a true bearing: frac in (-7.5, 7.5]."""
    k = round(deg / 15)
    return k % 24, deg - 15 * k


def favour(k, lead=0.6):
    """Evidence leaning toward step k by `lead` (a decent compass frame)."""
    ev = np.full(24, 1.5)
    ev[k % 24] += lead
    return ev


def off(a, b):
    return abs((a - b + 180) % 360 - 180)


def turn(t, start, n, prior, trusted=False, rate=1.5, wrong=None, t0=0.0):
    """Feed n reads of a turret turning `rate` deg per read from `start`; the evidence favours
    the true step (or step `wrong(i)` when given). Returns [(true deg, Track)]."""
    out = []
    for i in range(n):
        deg = (start + rate * i) % 360
        k, f = split(deg)
        ev = favour(k if wrong is None else wrong(i, k))
        out.append((deg, t.update(t0 + i * DT, f, ev, prior=prior if i == 0 else None, trusted=trusted)))
    return out


def test_a_trusted_prior_settles_briefly_then_reads():
    t = hb.StepTracker()
    k, f = split(100.0)
    rs = [t.update(i * DT, f, None, prior=101.0, trusted=True) for i in range(hb.SETTLE_READS)]
    assert all(r.deg is None for r in rs[:-1])
    assert rs[-1].state == "tracking" and off(rs[-1].deg, 100.0) < 1e-9


def test_tracks_across_steps_and_north():
    t = hb.StepTracker()
    for deg, r in turn(t, 352.0, 30, prior=352.0, trusted=True):
        if r.deg is not None:
            assert off(r.deg, deg) < 1e-9
    assert r.deg is not None and r.state in ("tracking", "switched")


def test_an_untrusted_prior_waits_for_agreeing_views():
    t = hb.StepTracker()
    seen = turn(t, 200.0, 12, prior=203.0)
    assert seen[0][1].deg is None and seen[1][1].deg is None
    shown = [(d, r) for d, r in seen if r.deg is not None]
    assert shown and all(off(r.deg, d) < 1e-9 for d, r in shown)


def test_no_evidence_never_confirms_an_untrusted_guess():
    # review 2026-10-11: with no compass digits, "nothing contradicts it" showed a 12 deg-off guess
    t = hb.StepTracker()
    k, f = split(200.0)
    rs = [t.update(i * DT, f, None, prior=212.0) for i in range(40)]
    assert all(r.deg is None for r in rs)
    assert "turn" in rs[-1].why


def test_a_wrong_prior_is_corrected_and_never_shown():
    # the target is 12 deg away: the prior lands on the wrong step; the compass favours the right one
    t = hb.StepTracker()
    seen = turn(t, 200.0, 40, prior=212.0)
    shown = [(d, r) for d, r in seen if r.deg is not None]
    assert shown and all(off(r.deg, d) < 1e-9 for d, r in shown)


def test_one_confident_wrong_frame_does_not_switch():
    t = hb.StepTracker()
    seen = turn(t, 90.0, 30, prior=90.0, trusted=True,
                wrong=lambda i, k: k + 3 if i == 10 else k)
    assert all(off(r.deg, d) < 1e-9 for d, r in seen if r.deg is not None)
    assert not any(r.state == "switched" for _, r in seen)


def test_identical_wrong_frames_on_a_still_turret_do_not_switch():
    # review 2026-10-11: a misread view repeats every read while the turret holds still; 55 of 65
    # such views switched a right track within 1-2 s
    t = hb.StepTracker()
    turn(t, 90.0, 10, prior=90.0, trusted=True)               # settled on the right step
    truth = (90.0 + 1.5 * 9) % 360
    k, f = split(truth)
    for i in range(80):                                        # 20 s, turret still, compass wrong
        r = t.update((10 + i) * DT, f, favour(k + 1, lead=1.5), prior=None)
        assert r.deg is not None and off(r.deg, truth) < 1e-9


def test_a_burst_of_two_wrong_frames_after_a_gap_does_not_switch():
    t = hb.StepTracker()
    seen = turn(t, 50.0, 20, prior=50.0, trusted=True,
                wrong=lambda i, k: k + 1 if i in (1, 2) else k)
    assert not any(r.state == "switched" for _, r in seen)
    assert all(off(r.deg, d) < 1e-9 for d, r in seen if r.deg is not None)


def test_a_sustained_contradiction_switches_a_tracked_step():
    t = hb.StepTracker()
    seen = turn(t, 45.0, 40, prior=60.0, trusted=True)        # anchored 15 deg wrong, turning
    assert any(r.state == "switched" for _, r in seen)
    d, r = seen[-1]
    assert off(r.deg, d) < 1e-9


def test_a_contradicted_trusted_prior_keeps_settling():
    t = hb.StepTracker()
    seen = turn(t, 45.0, 3, prior=60.0, trusted=True)          # the evidence says another step
    assert all(r.deg is None for _, r in seen)


def test_no_evidence_tracks_and_never_switches():
    t = hb.StepTracker()
    k, f = split(10.0)
    rs = [t.update(i * DT, f, None, prior=10.0, trusted=True) for i in range(20)]
    for r in rs[hb.SETTLE_READS - 1:]:
        assert r.state == "tracking" and off(r.deg, 10.0) < 1e-9


def test_a_gap_starts_settling_again():
    t = hb.StepTracker()
    k, f = split(300.0)
    for i in range(hb.SETTLE_READS):
        t.update(i * DT, f, None, prior=300.0, trusted=True)
    r = t.update(10.0, f, None, prior=300.0)                   # the map was open
    assert r.state == "settling" and r.deg is None


def test_no_track_and_no_prior_is_lost_with_a_reason():
    r = hb.StepTracker().update(0.0, 2.0, None, prior=None)
    assert r.deg is None and r.state == "lost" and "near" in r.why


def test_recorder_round_trips_through_replay(tmp_path):
    import glob
    import os
    import cv2
    here = os.path.dirname(os.path.abspath(__file__))
    tapes = sorted(glob.glob(os.path.join(here, "test_shots", "bearing_check", "b*.png")))[:3]
    rec = hb.Recorder(str(tmp_path))
    for i, p in enumerate(tapes):
        rec.save({"tape": cv2.imread(p), "box": None}, {"t": i * DT, "prior": 0.0, "trusted": False,
                                                       "deg": None, "state": "settling"})
    out = hb.replay(str(tmp_path))
    assert len(out) == 3 and all(isinstance(tr, hb.Track) for _, tr in out)
    assert out[0][0]["i"] == 0 and out[0][0]["state"] == "settling"


def _session(tmp_path, n=4, missing=None, notape=None):
    import glob
    import os
    import cv2
    here = os.path.dirname(os.path.abspath(__file__))
    tapes = sorted(glob.glob(os.path.join(here, "test_shots", "bearing_check", "b*.png")))[:n]
    rec = hb.Recorder(str(tmp_path))
    for i, p in enumerate(tapes):
        rec.save({"tape": cv2.imread(p), "box": None},
                 {"t": i * DT, "prior": 100.0, "trusted": False, "deg": None,
                  "state": "notape" if i == notape else "settling"})
    if missing is not None:
        os.remove(os.path.join(str(tmp_path), f"{missing:05d}_tape.png"))
    return tmp_path


def test_replay_survives_a_missing_tape_image(tmp_path):
    out = hb.replay(str(_session(tmp_path, missing=1)))
    assert len(out) == 4 and out[1][1].deg is None and "missing" in out[1][1].why


def test_replay_can_start_from_a_wrong_or_untrusted_guess(tmp_path):
    seen = []
    real = hb.StepTracker.update

    def spy(self, now, frac, ev, prior, trusted=False, proof=None):
        seen.append((prior, trusted))
        return real(self, now, frac, ev, prior, trusted, proof)
    hb.StepTracker.update = spy
    try:
        hb.replay(str(_session(tmp_path)), prior_offset=15.0, trust=False)
    finally:
        hb.StepTracker.update = real
    assert seen and all(p == 115.0 and t is False for p, t in seen)


def test_a_recording_stops_at_its_size_cap(tmp_path):
    import numpy as np
    rec = hb.Recorder(str(tmp_path), max_bytes=50_000)
    img = (np.random.default_rng(0).random((90, 720, 3)) * 255).astype(np.uint8)   # ~190 kB as png
    for i in range(5):
        rec.save({"tape": img}, {"t": i})
    assert rec.full
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1                                  # the first, then nothing past the cap
