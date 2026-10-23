"""Read the SPH-2's elevation off the gun view's MIL scale -- the sliding scale on the
right of the sight, "50 mils", "60 mils", ... -- to a hundredth of a mil. artillery.py
polls it, so the window can say how far to turn the elevation and log what was fired.

    python hud_mil.py shot.png [...]        # read whole 2560x1440 frames
    python hud_mil.py --strip strip.png     # read a BOX crop
    python hud_mil.py --build               # learn the digits from test_shots/mil

Measured 2026-09-30 on a 90 s sweep of the whole range (USER, 775 frames):
  - Ticks are a 27x5 px arm (frame x 1618-1644) crossed by a 4 px bar (x 1641-1644),
    one every 10 mils, EXACTLY 176 px apart. The value under the sight is on the
    dashed centre line, frame rows 719-720 (REF 719.5): a tick above it is a lower mil.
    Reading: label + (REF - tick row) / 176 x 10.
  - Strokes are white blended ~75% over the scene: ~200 over dark ground, ~249 over
    ~230 sky. So the arm is found by its edges -- its core rows against bands just clear
    of it, the median along the arm -- and every tick at once (a comb of 176 px), which
    a single tick lost in foliage cannot pull.
  - The scale scrolls at sub-pixel positions: an arm is 4-6 rows by anti-aliasing.
  - Labels are "<n> mils", with a thousands comma from 1,000 up, digits 15 px tall,
    and not at one x ("1,040" starts left of "30"). Ticks near the top and bottom of the
    sight's window lose their labels first; text over bright sky can vanish. So ONE
    clean label per frame is enough (the rest are it +-10 per tick), and labels that
    disagree are a reason, not a reading.
  - The sweep read 715 of 775 frames and never jumped between neighbours; the misses
    are labels hidden by clutter.

Nothing here guesses: a frame it cannot read is a reason, not a reading.
"""
import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "test_shots", "mil")
GLYPHS = os.path.join(HERE, "mil_glyphs.npz")

BOX = (1600, 360, 200, 720)     # x, y, w, h on the frame: the scale and its labels
SPACING = 176.0                 # px per 10 mils
REF = 719.5                     # frame row of the sight's centre line
ARM = (20, 38)                  # BOX x of the arm, clear of the bar
LABEL_X = (45, 200)             # BOX x of the labels
GW, GH = 12, 18                 # a glyph is compared at this size
TOOTH_MIN = 40                  # comb score of a tick whose label is worth reading: was 60 until
                                # live 2026-10-13, when ticks over an overcast sky scored 46-54 and
                                # their clean labels went unread (the rest over reeds) -- 2 of 5
                                # frames, every auto-shot stopped; 40 reads all 5, no held-out change
GLYPH_ERR = 0.25                # mean abs difference from the best template
GLYPH_MARGIN = 0.04             # ...and how much better than the second best


@dataclass
class Mil:
    mil: float                  # under the sight's centre line
    labels: int                 # how many labels agreed on it


def _row_score(strip):
    """Per BOX row y: how much rows y..y+4 look like a tick arm."""
    v = strip.max(axis=2).astype(float)
    p = np.median(v[:, ARM[0]:ARM[1]], axis=1)
    n = len(p)
    s = np.full(n, -1e9)
    if n < 20:
        return s
    core = (p[1:-3] + p[2:-2] + p[3:-1]) / 3          # core[y-1] = mean p[y+1..y+3]
    for y in range(5, n - 10):
        c = core[y]
        s[y] = min(c - p[y - 4:y - 1].mean(), c - p[y + 6:y + 9].mean())
    return s


def _comb(strip):
    """(phase, [(tooth index, BOX row of its arm centre, score)]) of the best 176 px comb."""
    s = _row_score(strip)
    best = None
    for ph in range(int(SPACING)):
        ys = np.arange(ph, len(s) - 6, SPACING).astype(int)
        sc = np.sort(s[ys])
        val = sc[len(sc) // 2:].sum()                  # the best half (clipped ends, clutter)
        if best is None or val > best[0]:
            best = (val, ph, ys)
    _, ph, ys = best
    return ph, [(k, int(y) + 2, s[y]) for k, y in enumerate(ys)]


def _label_mask(strip, c):
    roi = strip[c - 14:c + 15, LABEL_X[0]:LABEL_X[1]]
    v = roi.max(axis=2)
    th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, np.ones((11, 11), np.uint8))
    neutral = (roi.max(axis=2).astype(int) - roi.min(axis=2)) <= 30
    t, _ = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((th >= max(t, 25)) & neutral).astype(np.uint8)


def _glyphs(mask):
    """The number's glyph bitmaps left to right: digit-height ones and the comma, up to
    the 'm' of 'mils' (x-height, wide). None when the label is not clean.

    Live 2026-10-14 (recorded strips, 05:46 session) most clean-looking labels it refused had two
    digits run together in the mask ("50" one blob, 20-22 px wide; a digit is 9-12 px on a 12 px
    pitch) or a speck stuck on a digit's top (17-18 px tall, a digit is 15): a blob of whole digit
    widths on the row is cut at its thinnest columns, and a digit is trimmed to the row."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = sorted(((st[i, 0], st[i, 1], st[i, 2], st[i, 3], i) for i in range(1, n)
                    if st[i, 4] >= 4), key=lambda c: c[0])
    digit = lambda w, h: 13 <= h <= 16 and 3 <= w <= 12       # noqa: E731
    tops = sorted(y for x, y, w, h, i in comps if digit(w, h))
    if not tops:
        return None
    top = tops[len(tops) // 2]                           # the digits' row: they all start on it
    hs = sorted(h for x, y, w, h, i in comps if digit(w, h) and abs(y - top) <= 1)
    dh = hs[len(hs) // 2] if hs else 15                  # the digits' height
    out = []
    for x, y, w, h, i in comps:
        if 9 <= h <= 12 and 13 <= w <= 18 and abs(y + h - (top + dh)) <= 3:   # the 'm', on the baseline
            break
        g = (lab[y:y + h, x:x + w] == i).astype(np.float32)
        on_row = y <= top + 1 and y + h >= top + dh - 1 and y >= top - 5 and y + h <= top + dh + 5
        if on_row and h > dh:                            # a speck stuck on top or below: the row only
            g = g[top - y:top - y + dh]
            h = g.shape[0]
        if digit(w, h) and on_row:
            out.append(("d", g))
        elif on_row and 15 <= w <= 37 and h >= 13:       # 2-3 digits run together
            # two up to 24 px ("10" is 17: a "1" is 5-6 px wide, the rest 9-12), three above; each
            # cut is the emptiest column near where it should fall
            k = 2 if w <= 24 else 3
            col = g.sum(axis=0)
            cuts = [0]
            for j in range(1, k):
                lo, hi = max(cuts[-1] + 4, int(w * j / k) - 5), min(w - 4, int(w * j / k) + 6)
                if lo >= hi:
                    return None
                cuts.append(lo + int(np.argmin(col[lo:hi])))
            cuts.append(w)
            for a, b in zip(cuts, cuts[1:]):
                piece = g[:, a:b]
                xs = np.nonzero(piece.sum(axis=0))[0]
                if len(xs) < 3:
                    return None
                out.append(("d", piece[:, xs[0]:xs[-1] + 1]))
        elif 2 <= w <= 6 and 3 <= h <= 7 and out and top + 9 <= y + h <= top + 20:   # 5-6 px over clutter
            out.append((",", g))
        elif y + h < top + 2 or y > top + 14 or st[i, 4] <= 8 or (not out and h <= 9):
            # a speck off the digits' row, or a tiny one: the scene behind, not the label (live
            # 2026-10-14: one dot left of a clean "1,280 mil" lost the label). Anything bigger on the
            # row still refuses it -- skipping it read a "1,280" whose "1" was clutter as 280 -- except a
            # short one before the number (the tick's end, 4x3 px: no "1" is that short).
            continue
        else:
            return None                                  # clutter touching the number
    else:
        return None                                      # no 'm': the label is cut or hidden
    if not out or out[0][0] != "d" or out[-1][0] != "d":
        return None
    return [(kind, cv2.resize(g, (GW, GH), interpolation=cv2.INTER_AREA)) for kind, g in out]


def _labels(strip):
    """[(tooth index, BOX row, glyph bitmaps)] for every tick with a clean label."""
    _, teeth = _comb(strip)
    out = []
    for k, c, score in teeth:
        if score < TOOTH_MIN or not 14 <= c < strip.shape[0] - 14:
            continue
        gs = _glyphs(_label_mask(strip, c))
        if gs:
            out.append((k, c, gs))
    return teeth, out


class Templates:
    """Every learned glyph, by character: a glyph is the character whose NEAREST example is nearest
    (2026-10-14: one mean template per digit, learned from 40 sweep frames -- two of them an "8" --
    refused clean labels over clutter; averaged over the recordings' 4000 glyphs the means blurred
    and read less. The compass box and the tape's labels went the same way, hud_compass/bearing_labels_nn)."""

    def __init__(self, path=GLYPHS):
        with np.load(path) as z:
            X, Y = z["X"].astype(np.float32), z["Y"]
        self.t = {ch: X[Y == ch] for ch in sorted(set(Y.tolist())) if ch != ","}

    def number(self, gs):
        chars = []
        for kind, g in gs:
            if kind == ",":                 # known by its size and place, not its shape: a 2-4 px
                chars.append(",")           # blob's shape varies with what is behind it (live
                continue                    # 2026-10-12: "1,020" lost to a comma at 0.274)
            best = sorted((float(np.abs(t - g).mean(axis=(1, 2)).min()), ch) for ch, t in self.t.items())
            if best[0][0] > GLYPH_ERR or best[1][0] - best[0][0] < GLYPH_MARGIN:
                return None
            chars.append(best[0][1])
        s = "".join(chars)
        if "," in s and not (len(s) == 5 and s[1] == ","):
            return None
        if not s.isdigit() and "," not in s:
            return None
        v = int(s.replace(",", ""))
        return v if v % 10 == 0 else None


_templates = None


def read(strip, templates=None):
    """(Mil, "") or (None, why) for a BOX crop of the frame."""
    global _templates
    if templates is None:
        if _templates is None:
            _templates = Templates()
        templates = _templates
    teeth, labels = _labels(strip)
    if not labels:
        return None, "no MIL scale (map open, not in the gun seat, or its labels hidden)"
    votes = {}
    for k, c, gs in labels:
        v = templates.number(gs)
        if v is not None:
            votes.setdefault(v - 10 * k, []).append(k)
    if not votes:
        return None, "the MIL labels are not readable"
    ranked = sorted(votes.items(), key=lambda kv: -len(kv[1]))
    if len(ranked) > 1 and len(ranked[1][1]) >= len(ranked[0][1]):
        return None, "the MIL labels disagree"
    base, ks = ranked[0]
    row0 = teeth[0][1] + BOX[1]
    return Mil(base + (REF - row0) / SPACING * 10, len(ks)), ""


def build(sample_dir=SAMPLES, out=GLYPHS):
    """Learn the digit templates from samples named mil<reading>[_anything].png (BOX
    crops): every clean label's value follows from the reading and its tick's row.
    Returns (glyphs per character, samples used)."""
    per = {}
    used = 0
    for p in sorted(glob.glob(os.path.join(sample_dir, "mil*.png"))):
        m = re.match(r"mil(\d+(?:\.\d+)?)", os.path.basename(p))
        if not m:
            continue
        value = float(m.group(1))
        strip = cv2.imread(p)
        used += 1
        _, labels = _labels(strip)
        for k, c, gs in labels:
            label = value - (REF - (c + BOX[1])) / SPACING * 10
            text = f"{int(round(label / 10) * 10):,}"
            if len(text) == len(gs):
                for ch, (_, g) in zip(text, gs):
                    per.setdefault(ch, []).append(g)
    chars = sorted(per)
    np.savez_compressed(out, X=np.concatenate([np.asarray(per[k], np.float16) for k in chars]),
                        Y=np.array([k for k in chars for _ in per[k]]))
    return {k: len(v) for k, v in sorted(per.items())}, used


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--strip", action="store_true", help="the images are already BOX crops")
    ap.add_argument("--build", action="store_true", help=f"learn the digits from {SAMPLES}")
    args = ap.parse_args(argv)
    if args.build:
        counts, used = build()
        print(f"{GLYPHS}: {counts} from {used} samples")
        return 0
    x, y, w, h = BOX
    for p in args.images:
        im = cv2.imread(p)
        r, why = read(im if args.strip else im[y:y + h, x:x + w])
        print(f"{p}: " + (why if r is None else f"{r.mil:.2f} mil ({r.labels} labels)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
