"""Read the top compass box -- "353 N", the whole-degree bearing in the box at the top centre
of the game screen -- to settle which 15-degree step the bearing tape is on.

    python hud_compass.py --record --start 352        # spin the hull-fixed turret through a full circle
    python hud_compass.py --build                     # learn the digits from test_shots/compass
    python hud_compass.py shot.png [...]              # read whole 2560x1440 frames

The bearing tape (hud_bearing.py) is exact to a tenth of a degree but only says WHERE in a
15-degree step; the step comes from a guess, and a wrong guess stays wrong (USER live
2026-09-30: compass 352, window -16). This box is the game's own whole-degree number. It reads
1.28 degrees LESS than the tape (hud_bearing's docstring), so `near_tape()` hands the tape
reader the box's number plus that -- good to about 1 degree, far inside the 7.5 it needs.

The digits are light with a dark outline over whatever sky or ground is behind them, so each of
the three cells is high-passed (the background is smooth, the strokes are not), normalised,
and matched against every stored sample of each digit (nearest neighbour: one mean template per
digit scored 43% of whole numbers on 350 frames, the nearest neighbour 85%, because the
background behind the digits changes from sky to trees to ground). The samples come from
frames whose bearing is known: a `--record` run labels them from the tape itself, read near a
start bearing you give and then near the hull's heading plus the turret angle, so the turret
may be turned at any pace.

The real use is `totals()`: the tape gives a bearing up to a 15-degree step, so there are 24
candidates, each with a known box number, and totals() scores how well the box's digits fit each
one. hud_bearing.StepTracker adds those scores up over many reads (2026-10-11: one read deciding
alone was confidently wrong 12% of the time). A wrong units digit cannot change which candidate
fits best, only a wrong tens or hundreds digit can.

Nothing here guesses: a box it cannot read is a reason, not a reading.
"""
import argparse
import glob
import os
import re
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "test_shots", "compass")
DIGITS = os.path.join(HERE, "compass_digits.npz")

BOX = (1233, 28, 95, 38)        # x, y, w, h on the frame: the box with the digits inside it
DIGIT_ROWS = (9, 29)            # BOX rows of the digits (measured: frame y 38-55)
DIGIT_X0 = 14                   # BOX x where the first digit's cell starts (frame x 1247)
CELL_W = 11.7                   # px per digit cell: three of them span the number (frame x 1247-1282)
CELL_PAD = 1
TAPE_MINUS_BOX = 1.28           # degrees the tape reads above the box (hud_bearing.py)
BOX_BELOW_TAPE = 0.78          # the box reads the tape less this, rounded down (hud_bearing.py)
MIN_SCORE = 0.35                # a digit's best match must be at least this good...
MIN_MARGIN = 0.08               # ...and this much better than the runner-up

_store = None


def _cells(box):
    """The three digit cells of a BOX crop, each high-passed and normalised (zero mean, unit norm)."""
    g = box.max(axis=2).astype(np.float32) if box.ndim == 3 else box.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 3.0)
    out = []
    for i in range(3):
        x0 = int(round(DIGIT_X0 + CELL_W * i)) - CELL_PAD
        x1 = int(round(DIGIT_X0 + CELL_W * (i + 1))) + CELL_PAD
        p = g[DIGIT_ROWS[0]:DIGIT_ROWS[1], max(0, x0):x1]
        if p.shape[1] != 2 * CELL_PAD + int(round(CELL_W)):
            p = np.pad(p, ((0, 0), (0, 2 * CELL_PAD + int(round(CELL_W)) - p.shape[1])))
        p = p - p.mean()
        n = float(np.linalg.norm(p))
        out.append(p / n if n > 1e-6 else None)
    return out


def _load():
    """(samples N x D, digit of each, cell position of each), or None without a samples file."""
    global _store
    if _store is None:
        if not os.path.exists(DIGITS):
            _store = ()
        else:
            with np.load(DIGITS) as z:
                _store = (z["X"], z["Y"], z["P"])
    return _store or None


def has_samples():
    return _load() is not None


def digit_scores(cell, pos, store):
    """[best match of the cell to a stored sample of digit d, for d in 0..9] (-1 where none)."""
    X, Y, P = store
    sel = P == pos
    sims = X[sel] @ cell.ravel()
    ys = Y[sel]
    return [float(sims[ys == d].max()) if (ys == d).any() else -1.0 for d in range(10)]


def read(box):
    """(whole degrees 0-359, "") or (None, why) for a BOX crop: each digit is its nearest
    stored sample's. Clear frames only -- a busy background flips a digit now and then."""
    store = _load()
    if store is None:
        return None, "no digit samples yet: python hud_compass.py --record, then --build"
    digits = []
    for i, c in enumerate(_cells(box)):
        if c is None:
            return None, "the compass box is empty (map open?)"
        sc = digit_scores(c, i, store)
        order = sorted(range(10), key=lambda d: -sc[d])
        if sc[order[0]] < MIN_SCORE or sc[order[0]] - sc[order[1]] < MIN_MARGIN:
            return None, f"digit {i + 1} of the compass box is unclear"
        digits.append(order[0])
    deg = digits[0] * 100 + digits[1] * 10 + digits[2]
    if deg > 359:
        return None, f"the compass box read {deg}"
    return deg, ""


def expected_box(tape_deg):
    """The whole-degree number the box shows while the tape reads `tape_deg`: measured to read
    0.78-1.73 degrees below the tape, i.e. the tape less 0.78, rounded down."""
    return int(np.floor(tape_deg - BOX_BELOW_TAPE)) % 360


def totals(box, candidates):
    """([score of each candidate tape bearing], "") or (None, why): how well the box's three
    digits match each candidate's expected box number (sum of the three digit similarities,
    at most 3.0). hud_bearing.StepTracker adds these up over many reads."""
    store = _load()
    if store is None:
        return None, "no digit samples yet"
    cells = _cells(box)
    if any(c is None for c in cells):
        return None, "the compass box is empty"
    sc = [digit_scores(c, i, store) for i, c in enumerate(cells)]
    return [sum(sc[i][int(ch)] for i, ch in enumerate(f"{expected_box(cand):03d}"))
            for cand in candidates], ""


def choose(box, candidates):
    """(best tape bearing among `candidates`, margin over the runner-up, "") or (None, 0, why):
    the candidate whose expected box number's three digits match the box best."""
    tot, why = totals(box, candidates)
    if tot is None:
        return None, 0.0, why
    order = sorted(range(len(candidates)), key=lambda i: -tot[i])
    margin = tot[order[0]] - (tot[order[1]] if len(order) > 1 else -3.0)
    return candidates[order[0]], margin, ""


def near_tape(box):
    """(a bearing within ~1 degree of what the tape reads, "") or (None, why)."""
    deg, why = read(box)
    return (None, why) if deg is None else ((deg + TAPE_MINUS_BOX) % 360, "")


def grab_box(sct, game):
    x, y, w, h = BOX
    return cv2.cvtColor(np.asarray(sct.grab({"left": game["left"] + x, "top": game["top"] + y,
                                              "width": w, "height": h})), cv2.COLOR_BGRA2BGR)


def build(sample_dir=SAMPLES, out=DIGITS):
    """Store every labelled cell from samples named c<degrees>_<anything>.png (BOX crops), after
    dropping frames whose three digits all look like another number's (a wrong label: the tape
    lost its step while recording). Returns (samples per digit, frames used, frames dropped)."""
    frames = []
    for p in sorted(glob.glob(os.path.join(sample_dir, "c*.png"))):
        m = re.match(r"c(\d{3})_", os.path.basename(p))
        img = cv2.imread(p) if m else None
        if img is None or img.shape[0] < DIGIT_ROWS[1]:
            continue
        cells = _cells(img)
        if any(c is None for c in cells):
            continue
        frames.append((m.group(1), cells))
    X = np.array([c.ravel() for _, cs in frames for c in cs])
    Y = np.array([int(d) for lab, _ in frames for d in lab])
    P = np.array([i for _ in frames for i in range(3)])
    F = np.repeat(np.arange(len(frames)), 3)
    S = X @ X.T
    S[F[:, None] == F[None, :]] = -9                      # never score a frame against itself
    S[P[:, None] != P[None, :]] = -9                      # same cell position only
    pred = Y[S.argmax(axis=1)]
    wrong = {f for f in range(len(frames)) if not (pred[F == f] == Y[F == f]).any()}
    keep = ~np.isin(F, list(wrong))
    np.savez(out, X=X[keep].astype(np.float32), Y=Y[keep], P=P[keep])
    global _store
    _store = None
    counts = {d: int((Y[keep] == d).sum()) for d in range(10)}
    return counts, len(frames) - len(wrong), len(wrong)


def record(start_bearing, seconds, out_dir=SAMPLES, every_s=0.1):
    """Sample the compass box while the turret turns (gun view, hull parked), labelling each
    frame from the bearing tape. Frames within 0.3 degree of a whole-degree boundary are
    skipped: the box rounds a bearing that the tape only knows to its tenth."""
    import artillery
    from winkeys import dpi_aware
    dpi_aware()
    os.makedirs(out_dir, exist_ok=True)
    reader = artillery.Reader()
    hull, saved, skipped = None, 0, 0
    t_end = time.time() + seconds
    print(f"recording {seconds:.0f} s from tape bearing ~{start_bearing}: turn the turret through a full "
          f"circle, any pace; the hull must stay put. Ctrl+C stops.", flush=True)
    try:
        while time.time() < t_end:
            t0 = time.time()
            tilt, _ = reader.read_tilt()
            if tilt is not None:
                near = start_bearing if hull is None else (hull + tilt.turret_deg) % 360
                b, _ = reader.read_bearing(near)
                if b is not None:
                    if hull is None:
                        hull = (b.deg - tilt.turret_deg) % 360
                    box = grab_box(reader.sct, reader.game)
                    truth = b.deg - TAPE_MINUS_BOX
                    frac = truth - np.floor(truth)
                    if abs(frac - 0.5) < 0.12:         # too close to where the box flips to the next degree
                        skipped += 1
                    else:
                        cv2.imwrite(os.path.join(out_dir, f"c{int(round(truth)) % 360:03d}_{int(t0 * 1000)}.png"), box)
                        saved += 1
            time.sleep(max(0.0, every_s - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    print(f"saved {saved} frames ({skipped} skipped near a boundary) -> {out_dir}")
    return saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--start", type=float, help="the bearing tape now, within 7 degrees (with --record)")
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--build", action="store_true", help=f"learn the digits from {SAMPLES}")
    args = ap.parse_args(argv)
    if args.record:
        if args.start is None:
            ap.error("--record needs --start")
        record(args.start, args.seconds)
        return 0
    if args.build:
        counts, used, dropped = build()
        print(f"{DIGITS}: samples per digit {counts} from {used} frames ({dropped} dropped as mislabelled)")
        return 0
    x, y, w, h = BOX
    for p in args.images:
        im = cv2.imread(p)
        deg, why = read(im[y:y + h, x:x + w])
        print(f"{p}: " + (why if deg is None else f"{deg}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
