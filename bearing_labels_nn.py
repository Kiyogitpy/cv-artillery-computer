"""Learn the bearing tape's labels from --record-bearing sessions: bearing_labels_nn.npz, the
examples hud_bearing matches each label against (nearest neighbour).

    python bearing_labels_nn.py            # every session in test_shots/bearing_live

USER 2026-10-12: "never not have the right answer". The tape's labels are black; the September
templates (91 crops, one mean per label) read 225 as 120 and 240 as 300 on crisp frames -- 68% of
held-out reads right. Every recorded read carries its own truth: the minimap icon's hull heading
plus the tilt dial's turret angle gives the 15-degree step (independent of the tape), kept only
where the live tracker agreed (two sources). Each long tick's label then follows from that step.
Held out session by session (2026-10-12, 5056 reads of 8 sessions): nearest neighbour over these
examples read 88.8% right, and its most confident wrong pick led the next step by only 0.050 --
hud_bearing.LABEL_PROOF_MARGIN_NN sits well above that.
"""
import bisect
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

import hud_bearing
import minimap

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(HERE, "test_shots", "bearing_live")
PER_LABEL = 200                 # examples kept per label value (spread over the sessions)
EDGE_DEG = 4.0                  # truth nearer than this to a step's edge is not truth


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def truth_rows(sessions_dir=SESSIONS):
    """[(tape crop path, true k)] for every recorded read whose step the minimap icon plus the
    turret angle and the live tracker agree on."""
    out = []
    for d in sorted(glob.glob(os.path.join(sessions_dir, "*"))):
        lp, mp = os.path.join(d, "log.jsonl"), os.path.join(d, "minimap", "log.jsonl")
        if not (os.path.exists(lp) and os.path.exists(mp)):
            continue
        mm = [json.loads(line) for line in open(mp, encoding="utf-8")]
        ts = [r["t"] for r in mm]
        icons = {}
        for r in (json.loads(line) for line in open(lp, encoding="utf-8")):
            if r.get("turret") is None or r.get("deg") is None or r.get("state") not in ("tracking", "switched"):
                continue
            tape = os.path.join(d, f"{r['i']:05d}_tape.png")
            k = bisect.bisect_left(ts, r["t"])
            near = [j for j in (k - 1, k) if 0 <= j < len(mm) and abs(mm[j]["t"] - r["t"]) < 0.3]
            if not near or not os.path.exists(tape):
                continue
            j = min(near, key=lambda j: abs(mm[j]["t"] - r["t"]))
            if j not in icons:
                crop = cv2.imread(os.path.join(d, "minimap", f"{mm[j]['i']:05d}_minimap.png"))
                icons[j] = None if crop is None else minimap.icon_heading(crop)[0]
            if icons[j] is None:
                continue
            obs, _ = hud_bearing.observe(cv2.imread(tape))
            if obs is None:
                continue
            pred = (icons[j] + r["turret"]) % 360
            kk = round((pred - obs.frac) / 15) % 24
            if abs(_wrap(pred - (15 * kk + obs.frac))) > EDGE_DEG:
                continue
            if abs(_wrap(r["deg"] - (15 * kk + obs.frac))) >= 2.0:     # the tracker disagrees
                continue
            out.append((tape, kk))
    return out


def sample_rows(sample_dir=hud_bearing.SAMPLES):
    """[(tape crop path, true k)] for the September traverse's crops (b<bearing>_*.png): every
    direction, where today's sessions barely faced 15-60 and 165-210. test_shots/bearing_check
    stays out -- it is held out."""
    out = []
    for p in sorted(glob.glob(os.path.join(sample_dir, "b*.png"))):
        m = re.match(r"b(\d+(?:\.\d+)?)_", os.path.basename(p))
        obs, _ = hud_bearing.observe(cv2.imread(p)) if m else (None, "")
        if obs is not None:
            out.append((p, round((float(m.group(1)) - obs.frac) / 15) % 24))
    return out


def build(rows, out=hud_bearing.LABELS_NN, per_label=PER_LABEL):
    """Cut every long tick's label from the rows' tape crops, keep up to per_label examples per
    value (evenly spread), and save them. Returns {value: examples}."""
    per = {}
    for tape_path, k in rows:
        tape = cv2.imread(tape_path)
        _, longs, _ = hud_bearing._ticks(tape)
        if len(longs) < 2:
            continue
        ic = int(np.argmin(np.abs(longs - hud_bearing.CENTRE)))
        for i, x in enumerate(longs):
            p = hud_bearing._patch(tape, x + 1)
            if p is not None:
                per.setdefault((15 * (k + i - ic)) % 360, []).append(p.ravel())
    X, Y = [], []
    for v, ps in sorted(per.items()):
        pick = np.linspace(0, len(ps) - 1, min(per_label, len(ps))).round().astype(int)
        X += [ps[i] for i in pick]
        Y += [v] * len(pick)
    np.savez_compressed(out, X=np.array(X, np.float32), Y=np.array(Y, np.int16))
    hud_bearing._label_nn_store = None
    return {v: int((np.array(Y) == v).sum()) for v in sorted(per)}


def main():
    rows = truth_rows() + sample_rows()
    counts = build(rows)
    print(f"{hud_bearing.LABELS_NN}: {sum(counts.values())} examples from {len(rows)} reads: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
