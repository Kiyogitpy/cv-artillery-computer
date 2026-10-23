"""Read where the gun points off the gun view's bearing tape -- the scale across the
top of the sight, "225 240 255 ..." -- to a tenth of a degree. artillery.py polls it
next to the MIL scale.

    python hud_bearing.py --near 13 shot.png [...]   # read whole 2560x1440 frames
    python hud_bearing.py --build                    # learn the labels from test_shots/bearing

Measured 2026-09-30 on two full traverses (USER, 1383 frames, turret turning on flat
concrete):
  - A tick every 3 degrees, EXACTLY 22.2 px apart (7.4 px a degree), strokes 3 px wide
    on a baseline; every fifth tick is long and labelled with its bearing (a multiple of
    15). The bearing under the sight is at frame x 1280 (the tape's end brackets are
    symmetric about it); bearings grow to the right.
  - The TOP COMPASS BOX reads 1.28 deg LESS than this tape (23 frames: +0.78..+1.73,
    the spread of the box's whole degrees). The USER's ten shots of 2026-09-30 landed
    1.3 deg right of a target aimed by the box: this tape, not the box, is where the
    gun points.
  - The ticks are sharp in every scene (the fraction came out in 1379 of 1383 frames and
    never jumped). The labels are small and often lost in clutter: matched as whole
    labels (24 of them) they fix the 15-degree step in only about half the frames. So
    the step comes from `near` -- the target's azimuth, or the last reading -- which
    must be within 7.5 degrees; the labels only VETO a step they clearly contradict.

THE STEP (2026-10-11). Measured on footage each reader never saw: the labels pick the right
15-degree step 57% of the time, the top compass box 70% (confidently wrong 12%); a single read
deciding alone regularly knocked a right reading onto a wrong step. So artillery.py no longer
reads one frame: observe() gives each read's exact position inside the step plus both scores
for all 24 steps, and StepTracker follows the step from read to read (the turret turns at most
6.3 deg/s, 1.6 deg per poll), adds the scores up as fading votes -- once per VIEW, since a still
turret repeats one view and a misread view would repeat its error (review 2026-10-11) -- and moves
only when another step leads on SWITCH_MIN_READS counted reads in a row. After a gap it shows
nothing until it is sure: a trusted start (hull heading + turret) after SETTLE_READS uncontradicted
reads, a guess (the target's azimuth) only once CONFIRM_VIEWS views agree -- traverse a little.
Record a session with `--record-bearing` (artillery) and re-run it with `--replay`.

Nothing here guesses: a tape it cannot read is a reason, not a reading.
"""
import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "test_shots", "bearing")
LABELS = os.path.join(HERE, "bearing_labels.npz")
LABELS_NN = os.path.join(HERE, "bearing_labels_nn.npz")    # learned examples (bearing_labels_nn.py)

BOX = (920, 355, 720, 90)       # x, y, w, h on the frame: the tape and its labels
SPACING = 22.2                  # px per 3 degrees
PX_PER_DEG = SPACING / 3
CENTRE = 360.0                  # BOX x of the sight's centre (frame x 1280)
TICK_ROWS = (62, 71)            # BOX rows every tick crosses
LONG_ROWS = (49, 58)            # BOX rows only the long (labelled) ticks cross
LONG_MIN = 5                    # how much more the long ticks must show there than the short
LABEL_ROWS = (16, 35)
LABEL_HALF = 18
VETO = 0.15                     # labels preferring another step by this much refuse the reading
VETO_MIN = 0.4                  # ...when that step's own match is at least this good
LABEL_WEIGHT = 0.3              # tape labels count this much against the compass box (57% held out)
# USER 2026-10-12: the tape's labels are black -- over a bright sky they read clearly. Measured on
# 1366 reads of today's sessions whose step the minimap confirmed: a label pick leading the next
# step by LABEL_PROOF_MARGIN was never wrong, so it PROVES the step; PROOF_READS such reads in a row
# adopt it (StepTracker). Held out over 7345 reads of 8 sessions the most confident wrong pick led
# by 0.126 (a frame showing 270 and 285 plainly: test_shots/bearing_proof/*_trap_*), so 0.15.
LABEL_PROOF_MARGIN = 0.15
# With the learned examples (bearing_labels_nn.py) the scores are another scale: held out over 5056
# reads whose truth two sources agreed on, the most confident wrong pick led by 0.050; 70% of reads
# led by 0.10 and none of them was wrong.
LABEL_PROOF_MARGIN_NN = 0.12
PROOF_MIN_EXAMPLES = 50         # ...on labels with at least this many learned examples (see observe)
PROOF_READS = 2


@dataclass
class Bearing:
    deg: float                  # 0-360, clockwise from north, under the sight
    labels: str                 # "agree", "unseen" (too few/unclear), never "disagree"


def _col_score(tape, rows):
    """Per BOX column x: how much a 3 px stroke at x..x+2 stands out of its sides."""
    v = tape.max(axis=2).astype(float)
    p = np.median(v[rows[0]:rows[1]], axis=0)
    n = len(p)
    s = np.full(n, -1e9)
    for x in range(5, n - 8):
        core = p[x:x + 3].mean()
        s[x] = min(core - p[x - 4:x - 1].mean(), core - p[x + 4:x + 7].mean())
    return s


def _ticks(tape):
    """(every tick's x, the long ticks' x, long-tick strength) -- x of a stroke's left edge."""
    s = _col_score(tape, TICK_ROWS)
    best = None
    for ph in np.arange(0, SPACING, 0.25):
        xs = np.round(np.arange(ph, len(s) - 8, SPACING)).astype(int)
        xs = xs[xs >= 5]
        sc = np.sort(s[xs])
        val = sc[len(sc) // 3:].sum()
        if best is None or val > best[0]:
            best = (val, xs)
    xs = best[1]
    su = _col_score(tape, LONG_ROWS)
    pick = None
    for j in range(5):
        sel = xs[j::5]
        val = np.sort(su[sel])[len(sel) // 3:].sum()
        if pick is None or val > pick[0]:
            pick = (val, sel)
    longs = pick[1]
    shorts = np.setdiff1d(xs, longs)
    strength = float(np.median(su[longs]) - np.median(su[shorts])) if len(shorts) else 0.0
    return xs, longs, strength


def _patch(tape, x):
    """The label above a long tick at centre x, normalised: text bright, zero mean, unit norm."""
    x = int(round(x))
    if x - LABEL_HALF < 0 or x + LABEL_HALF + 1 > tape.shape[1]:
        return None
    p = tape[LABEL_ROWS[0]:LABEL_ROWS[1], x - LABEL_HALF:x + LABEL_HALF + 1].max(axis=2).astype(np.float32)
    med = np.median(p)
    if med - p.min() > p.max() - med:
        p = -p                                        # dark text over sky: make the text bright
    p = p - p.mean()
    n = float(np.linalg.norm(p))
    return p / n if n > 1e-6 else None


_labels = None


def _templates():
    global _labels
    if _labels is None:
        if not os.path.exists(LABELS):
            _labels = {}
        else:
            with np.load(LABELS) as z:
                _labels = {int(k): z[k] for k in z.files}
    return _labels


_label_nn_store = None


def _label_nn():
    """(examples N x D, value of each) from bearing_labels_nn.npz, or None without it."""
    global _label_nn_store
    if _label_nn_store is None:
        if not os.path.exists(LABELS_NN):
            _label_nn_store = ()
        else:
            with np.load(LABELS_NN) as z:
                _label_nn_store = (z["X"].astype(np.float32), z["Y"].astype(int))
    return _label_nn_store or None


def _label_scores(tape, longs):
    """{value of the first long tick: mean label match}, or {} without templates or labels. With
    bearing_labels_nn.npz each label scores its best match among the learned examples of each value
    (bearing_labels_nn.py: 88.8% of held-out reads right against the templates' 68%)."""
    ps = [p for p in (_patch(tape, x + 1) for x in longs)]
    if sum(p is not None for p in ps) < 2:
        return {}
    nn = _label_nn()
    if nn is not None:
        X, Y = nn
        values = range(0, 360, 15)
        best = [None if p is None else {v: float(s[Y == v].max()) for v in values if (Y == v).any()}
                for p, s in ((p, None if p is None else X @ p.ravel()) for p in ps)]
        out = {}
        for base in values:
            sc = [b[(base + 15 * i) % 360] for i, b in enumerate(best)
                  if b is not None and (base + 15 * i) % 360 in b]
            if sc:
                out[base] = sum(sc) / len(sc)
        return out
    T = _templates()
    if not T:
        return {}
    out = {}
    for base in range(0, 360, 15):
        sc = [float((p * T[(base + 15 * i) % 360]).sum()) for i, p in enumerate(ps)
              if p is not None and (base + 15 * i) % 360 in T]
        if sc:
            out[base] = sum(sc) / len(sc)
    return out


def fraction(tape):
    """Degrees past a 15-degree step the tape reads (0 <= frac < 15), or None when the tape is not
    there. The bearing is 15*k + frac for some whole k -- which k is what `near` or the top
    compass box (hud_compass.py) decides."""
    xs, longs, strength = _ticks(tape)
    if len(longs) < 2 or strength < LONG_MIN:
        return None
    xl = float(longs[np.argmin(np.abs(longs - CENTRE))]) + 1.0
    return ((CENTRE - xl) / PX_PER_DEG) % 15


@dataclass
class TapeObs:
    """One read of the tape as evidence for every 15-degree step: the bearing is 15*k + frac
    for one k in 0..23, and `labels[k]` / `compass[k]` say how well the tape's labels and the
    top compass box fit that k. StepTracker decides k from these over many reads."""
    frac: float                 # degrees from the long tick nearest the sight to the sight
    labels: list = None         # tape-label score for each k, or None
    compass: list = None        # compass-box score for each k (hud_compass.totals), or None
    proof: int = None           # the k the labels pick by LABEL_PROOF_MARGIN, or None

    def candidates(self):
        return [(15 * k + self.frac) % 360 for k in range(24)]

    def evidence(self):
        if self.labels is None and self.compass is None:
            return None
        ev = np.zeros(24)
        if self.compass is not None:
            ev += np.asarray(self.compass, float)
        if self.labels is not None:
            ev += LABEL_WEIGHT * np.asarray(self.labels, float)
        return ev


def observe(tape):
    """(TapeObs, "") or (None, why) for a BOX crop: the position inside the step (exact, the
    ticks) and the labels' score for each step (weak alone). The compass box is the caller's."""
    xs, longs, strength = _ticks(tape)
    if len(longs) < 2 or strength < LONG_MIN:
        return None, "no bearing tape (map open, not in the gun seat, or it is hidden)"
    ic = int(np.argmin(np.abs(longs - CENTRE)))
    frac = (CENTRE - (float(longs[ic]) + 1.0)) / PX_PER_DEG
    scores = _label_scores(tape, longs)        # {label of the FIRST long tick: score}
    labels = ([scores.get((15 * k - 15 * ic) % 360, 0.0) for k in range(24)]
              if scores else None)
    proof = None
    if labels:
        order = np.argsort(labels)[::-1]
        nn = _label_nn()
        cut = LABEL_PROOF_MARGIN_NN if nn is not None else LABEL_PROOF_MARGIN
        if labels[order[0]] - labels[order[1]] >= cut:
            k = int(order[0])
            # the zero-wrong cut holds where it was measured: every label this step puts on the
            # tape needs PROOF_MIN_EXAMPLES learned examples, or the labels vote but never prove
            seen = [(15 * (k + i - ic)) % 360 for i in range(len(longs))]
            if nn is None or all((nn[1] == v).sum() >= PROOF_MIN_EXAMPLES for v in seen):
                proof = k
    return TapeObs(frac, labels, proof=proof), ""


# The step tracker (2026-10-11). Measured: the compass box picks the right step 70% of the time
# on footage it never saw, but is confidently WRONG 12% -- and it used to decide alone, on every
# read, so a right reading was regularly knocked onto a wrong step and then tracked there. The
# turret turns at most 6.3 deg/s, 1.6 deg per 0.25 s poll, so between reads the step follows from
# the last one; the evidence only ever moves it when it has added up.
TRACK_GAP_S = 1.0               # reads further apart than this start over (~6 deg of traverse)
VOTE_HALF_LIFE_S = 3.0          # evidence fades with this half-life
SWITCH_MARGIN = 1.5             # another step must lead the tracked one by this in the votes...
SWITCH_MIN_READS = 4            # ...on this many counted reads in a row, each preferring it itself
SETTLE_READS = 4                # reads a trusted start (hull heading + turret) waits, uncontradicted
CONFIRM_VIEWS = 3               # counted reads an untrusted start (the target's azimuth) needs...
CONFIRM_MARGIN = 0.5            # ...with its step leading every other by this in the votes
# A read's evidence counts only when the bearing moved this far since the last one that counted.
# Review 2026-10-11: on a still turret every frame is the same view, so a misread view repeats
# every read; counted each time, 55 of 65 such held-out views switched a right track in 1-2 s.
NEW_VIEW_DEG = 1.0
SETTLE_HINT_S = 2.0             # settling this long: ask for a little traverse (new views)


@dataclass
class Track:
    deg: float = None           # None while settling or lost
    state: str = "lost"         # "tracking", "switched", "settling" or "lost"
    why: str = ""


class StepTracker:
    """Follows the bearing tape's 15-degree step from read to read. Knows nothing about Windows.

    update() takes each read's position inside the step (`frac`), its evidence (24 scores, one
    per step k, or None) and a `prior` bearing for when there is no track. A `trusted` prior (the
    hull's heading plus the turret angle) is shown after SETTLE_READS reads unless the evidence
    favours another step; an untrusted one (the target's azimuth) only once CONFIRM_VIEWS distinct
    views agree by CONFIRM_MARGIN -- with no evidence it never is. Evidence counts once per view
    (NEW_VIEW_DEG), and the step moves only when another leads by SWITCH_MARGIN on
    SWITCH_MIN_READS counted reads in a row that each prefer it on their own."""

    def __init__(self):
        self.k = None               # the step: bearing = 15*k + frac
        self.frac = None
        self.t = None
        self.votes = np.zeros(24)   # votes[j]: evidence that the true step is self.k + j
        self.streak = 0
        self.settle = 0             # >0: trusted reads left; -1: until the evidence confirms
        self.views = 0              # counted reads since this start
        self.last_view = None       # the (unwrapped) bearing of the last counted read
        self.since = None           # when this start began
        self.switches = 0           # steps switched by the evidence, ever
        self.proof_last, self.proof_run = None, 0   # the labels' proven k, and reads in a row of it
        self.label_proven = False   # this track's step was proven by the labels

    def confirmed(self, views=CONFIRM_VIEWS, margin=CONFIRM_MARGIN):
        """The step is backed by the evidence: `views` views since this start, leading every other
        step by `margin`."""
        if self.k is None:
            return False
        return self.views >= views and self.votes[0] - np.delete(self.votes, 0).max() >= margin

    def update(self, now, frac, evidence, prior, trusted=False, proof=None):
        # a trusted prior replaces an untrusted start the evidence has not confirmed yet (review
        # 2026-10-12: after the map the first read came before the tilt dial, and a gun at rest
        # never confirms the target's azimuth)
        upgrade = trusted and prior is not None and self.settle < 0
        if self.t is not None and now - self.t <= TRACK_GAP_S and not upgrade:
            last = 15 * self.k + self.frac
            self.k = min((self.k - 1, self.k, self.k + 1), key=lambda k: abs(15 * k + frac - last))
            self.votes *= 0.5 ** ((now - self.t) / VOTE_HALF_LIFE_S)
        else:
            if prior is None:
                self.t = None
                return Track(None, "lost", "no bearing to read it near (no target yet)")
            self.k = round((prior - frac) / 15)
            self.votes = np.zeros(24)
            self.streak, self.views, self.last_view, self.since = 0, 0, None, now
            self.settle = SETTLE_READS if trusted else -1
            self.proof_last, self.proof_run, self.label_proven = None, 0, False
        self.frac, self.t = frac, now
        here = 15 * self.k + frac
        state = "tracking"
        if evidence is None:
            self.streak = 0
        elif self.last_view is None or abs(here - self.last_view) >= NEW_VIEW_DEG:
            self.last_view, self.views = here, self.views + 1
            ev = np.roll(np.asarray(evidence, float), -(self.k % 24))
            self.votes += ev
            j = int(np.argmax(self.votes))
            if j and self.votes[j] - self.votes[0] >= SWITCH_MARGIN and int(np.argmax(ev)) == j:
                self.streak += 1
                if self.streak >= SWITCH_MIN_READS:
                    shift = j if j <= 12 else j - 24
                    self.k += shift
                    self.last_view += 15 * shift
                    self.votes = np.roll(self.votes, -j)
                    self.streak, self.settle = 0, 0
                    self.switches += 1
                    self.label_proven = False
                    state = "switched"
            else:
                self.streak = 0
        if proof is None:
            self.proof_last, self.proof_run = None, 0
        else:
            self.proof_run = self.proof_run + 1 if proof == self.proof_last else 1
            self.proof_last = proof
            if self.proof_run >= PROOF_READS:   # the labels prove the step: take it, shown at once
                shift = (proof - self.k + 12) % 24 - 12
                if shift:
                    self.k += shift
                    if self.last_view is not None:
                        self.last_view += 15 * shift
                    self.votes = np.roll(self.votes, -shift)
                    self.switches += 1
                    state = "switched"
                self.streak, self.settle, self.label_proven = 0, 0, True
        if self.settle > 0:
            self.settle -= 1
            if self.settle == 0 and int(np.argmax(self.votes)) != 0:
                self.settle = 1                 # the votes say another step: keep waiting
        elif self.settle < 0:
            lead = self.votes[0] - np.delete(self.votes, 0).max()
            if self.views >= CONFIRM_VIEWS and lead >= CONFIRM_MARGIN:
                self.settle = 0
        if self.settle:
            why = ("settling: turn the turret a little to confirm the 15-degree step"
                   if now - self.since >= SETTLE_HINT_S else "settling: confirming the 15-degree step")
            return Track(None, "settling", why)
        return Track((15 * self.k + self.frac) % 360, state)


def _plain(o):
    """numpy numbers and arrays as JSON (a float32 icon heading crashed every minimap tick, live
    2026-10-12)."""
    if isinstance(o, (np.generic, np.ndarray)):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


class Recorder:
    """Saves each read during play -- the crops it was made from and what was decided -- so a
    session that went wrong can be gone through offline: the bearing tape's with replay() /
    --replay, the MIL scale's with `python hud_mil.py --strip <file>`."""

    def __init__(self, directory, max_bytes=2_000_000_000):
        self.dir, self.i = directory, 0
        self.max_bytes, self.bytes, self.full = max_bytes, 0, False   # ~0.9 GB an hour of play
        os.makedirs(directory, exist_ok=True)

    def save(self, images, entry):
        """images: {name: BGR crop or None}, each written as NNNNN_<name>.png; entry: the log line.
        Past max_bytes the session stops recording (once, with a warning) rather than fill the disk."""
        if self.full:
            return
        base = os.path.join(self.dir, f"{self.i:05d}")
        for name, img in images.items():
            if img is not None:
                if not cv2.imwrite(f"{base}_{name}.png", img):
                    # live 2026-10-14: a full disk raised here on every read, every tick
                    self.full = True
                    print(f"recording {self.dir}: could not write (disk full?) -- stopped", flush=True)
                    return
                self.bytes += os.path.getsize(f"{base}_{name}.png")
        line = json.dumps({"i": self.i, **entry}, default=_plain) + "\n"
        with open(os.path.join(self.dir, "log.jsonl"), "a", encoding="utf-8") as f:
            f.write(line)
        self.bytes += len(line)
        self.i += 1
        if self.bytes > self.max_bytes:
            self.full = True
            print(f"recording {self.dir}: {self.bytes / 1e9:.1f} GB, stopped (the cap)", flush=True)


def replay(directory, prior_offset=0.0, trust=None):
    """[(log entry, Track)] for a Recorder session, re-read from its crops by today's readers.
    prior_offset moves every logged prior (a guess that many degrees wrong); trust, when not None,
    replaces every logged trusted flag -- to stress the tracker with a recording that went well."""
    import hud_compass
    tracker, out = StepTracker(), []
    with open(os.path.join(directory, "log.jsonl"), encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    for e in entries:
        base = os.path.join(directory, f"{e['i']:05d}")
        if not os.path.exists(base + "_tape.png"):
            out.append((e, Track(None, "lost", "missing tape image")))
            continue
        obs, why = observe(cv2.imread(base + "_tape.png"))
        if obs is None:
            out.append((e, Track(None, "lost", why)))
            continue
        box = cv2.imread(base + "_box.png") if os.path.exists(base + "_box.png") else None
        if box is not None and hud_compass.has_samples():
            obs.compass, _ = hud_compass.totals(box, obs.candidates())
        prior = e.get("prior")
        prior = None if prior is None else (prior + prior_offset) % 360
        out.append((e, tracker.update(e["t"], obs.frac, obs.evidence(), prior,
                                      e.get("trusted", False) if trust is None else trust, obs.proof)))
    return out


def read(tape, near):
    """(Bearing, "") or (None, why) for a BOX crop; `near` is a bearing within 7.5 deg of
    the truth (the target's azimuth, or the last reading)."""
    if near is None:
        return None, "no bearing to read it near (no target yet)"
    xs, longs, strength = _ticks(tape)
    if len(longs) < 2 or strength < LONG_MIN:
        return None, "no bearing tape (map open, not in the gun seat, or it is hidden)"
    xl = float(longs[np.argmin(np.abs(longs - CENTRE))]) + 1.0
    frac = (CENTRE - xl) / PX_PER_DEG                  # degrees from that long tick to the centre
    k = round((near - frac) / 15)
    deg = (15 * k + frac) % 360
    scores = _label_scores(tape, longs)
    verdict = "unseen"
    if scores:
        x0 = float(longs[0]) + 1.0
        mine = int(round((deg - (CENTRE - x0) / PX_PER_DEG) / 15) * 15) % 360
        best = max(scores, key=scores.get)
        if best != mine and scores[best] >= VETO_MIN and scores[best] - scores.get(mine, -1) >= VETO:
            other = (deg + (best - mine)) % 360
            return None, (f"the tape's labels say about {other:.0f} deg, not {deg:.0f}: "
                          f"turn onto the target by the compass first")
        if best == mine:
            verdict = "agree"
    return Bearing(deg, verdict), ""


def build(sample_dir=SAMPLES, out=LABELS):
    """Learn the 24 labels from samples named b<bearing>[_anything].png (BOX crops).
    Returns ({label: samples}, files used)."""
    per = {}
    used = 0
    for p in sorted(glob.glob(os.path.join(sample_dir, "b*.png"))):
        m = re.match(r"b(\d+(?:\.\d+)?)", os.path.basename(p))
        if not m:
            continue
        deg = float(m.group(1))
        tape = cv2.imread(p)
        used += 1
        _, longs, strength = _ticks(tape)
        if strength < LONG_MIN:
            continue
        for x in longs:
            xc = float(x) + 1.0
            lab = int(round((deg + (xc - CENTRE) / PX_PER_DEG) / 15) * 15) % 360
            q = _patch(tape, xc)
            if q is not None:
                per.setdefault(lab, []).append(q)
    T = {k: np.mean(v, axis=0) for k, v in per.items()}
    np.savez(out, **{str(k): v / np.linalg.norm(v) for k, v in T.items()})
    global _labels
    _labels = None
    return {k: len(v) for k, v in sorted(per.items())}, used


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--near", type=float, help="a bearing within 7.5 deg of the truth")
    ap.add_argument("--strip", action="store_true", help="the images are already BOX crops")
    ap.add_argument("--build", action="store_true", help=f"learn the labels from {SAMPLES}")
    ap.add_argument("--replay", help="a test_shots/bearing_live/<session> folder to re-run")
    ap.add_argument("--prior-offset", type=float, default=0.0,
                    help="replay: move every prior by this many degrees (a wrong guess)")
    ap.add_argument("--no-trust", action="store_true", help="replay: treat every prior as untrusted")
    args = ap.parse_args(argv)
    if args.replay:
        rows = replay(args.replay, args.prior_offset, False if args.no_trust else None)
        differ = 0
        for e, tr in rows:
            same = (e.get("deg") is None) == (tr.deg is None) and (
                tr.deg is None or abs((tr.deg - e["deg"] + 180) % 360 - 180) < 0.05)
            differ += not same
            print(f"{e['t']:9.2f}  logged {e.get('deg')!s:>8} {e.get('state', ''):9}  "
                  f"replayed {('%.2f' % tr.deg) if tr.deg is not None else '-':>8} {tr.state:9}"
                  + ("" if same else "   <-- differs"))
        shown = sum(tr.deg is not None for _, tr in rows)
        print(f"{len(rows)} reads, {shown} shown, {len(rows) - shown} settling/lost, {differ} differ")
        return 0
    if args.build:
        counts, used = build()
        print(f"{LABELS}: {counts} from {used} samples")
        return 0
    x, y, w, h = BOX
    for p in args.images:
        im = cv2.imread(p)
        b, why = read(im if args.strip else im[y:y + h, x:x + w], args.near)
        print(f"{p}: " + (why if b is None else f"{b.deg:.2f} deg (labels {b.labels})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
