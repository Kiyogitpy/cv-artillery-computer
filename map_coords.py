"""Read the map cursor's coordinates -- the "x98.99" / "y50.77" labels the game draws
beside the map crosshair -- off a 1440p frame.

    python map_coords.py shot.jpg --at 979,978      # read the labels of that crosshair
    python map_coords.py --build                    # rebuild glyphs.npz from test_shots/map

Units are the game's map units, 100 m each (the calculator in wardogs/ uses the same).

Layout, measured 2026-09-29 over six captures at two zoom levels (+-1px): relative to
where the crosshair lines cross (the cursor), the "x" label's letter sits at (+42,
-19) and the "y" label's at (+9, -93), and neither moved with the cursor near the
map's right edge. Near the TOP edge the y label would leave the map; what the game
does there has not been captured yet -- a failed read says so rather than guessing.

Reading one label:
  1. Find its letter by template match within SEARCH px of where the cursor says it
     is, so a small offset between the OS cursor and the drawn crosshair is absorbed.
  2. Cut the digits after it apart at empty columns of a white-text mask (V>180,
     S<70: the text is ~250 grey with a dark outline; coloured icons fall out).
  3. Match each digit's SHAPE (its text mask) against exemplars cut from the
     labelled captures (glyphs.npz, built by --build): nearest by normalised
     correlation, allowing a 1px shift. Left-one-capture-out over the six captures
     that read 49/49 digits with the closest call won by 0.104; matching grey
     levels instead read 48/49 (a 0 taken for a 9). A digit that does not beat the
     next-best DIFFERENT digit by MIN_MARGIN is unreadable, never a guess.
  4. Accept only <digits>.<2 digits>; anything after the second decimal (an icon's
     small white "4" sat right there in one capture) is ignored.
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
GLYPHS = os.path.join(HERE, "glyphs.npz")
SAMPLES = os.path.join(HERE, "test_shots", "map")

LETTER_AT = {"x": (42, -19), "y": (9, -93)}   # letter's top-left, from the crossing
SEARCH = 12             # px either way the letter is looked for
TEXT_V, TEXT_S = 180, 70
TEXT_V_OVER, TEXT_V_MAX = 15, 230   # over a light map: above its brightest tenth by this, at most this
LETTER_BOX = (-3, -6, 14, 20)     # template patch around a letter: dx, dy, w, h
DIGIT_ROWS = (-6, 12)             # digit band from the letter's top: glyphs are 15 tall, 4px higher
DIGIT_W = 13                      # digit patch width, centred on the glyph
MIN_SCORE = 0.40                  # a digit match weaker than this is "unreadable"
MIN_MARGIN = 0.04                 # ...and so is one this close to a different digit
MIN_LETTER = 0.55

# The grab needed around the cursor to hold both labels, with the search margin.
GRAB = (-5, -93 - 6 - SEARCH - 2, 160, 93 + 6 + SEARCH + 2 + 6)   # dx, dy, w, h


@dataclass
class Reading:
    x: float | None
    y: float | None
    why: str = ""          # empty when both read

    @property
    def ok(self):
        return self.x is not None and self.y is not None


def _gray_hsv(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _patch(img, x, y, w, h):
    """img[y:y+h, x:x+w], zero-padded where it leaves the image."""
    out = np.zeros((h, w), img.dtype)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 < x1 and y0 < y1:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = img[y0:y1, x0:x1]
    return out


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _text_mask(hsv, strict=False):
    """The white text: bright (above TEXT_V) and colourless. `strict`, the retry of a label that
    did not read: bright also means above the background itself (live 2026-10-12: a red-held zone
    tints the map pink, brightness ~180 with its brightest tenth ~211, and 500 background pixels
    passed a fixed 180). Not the first try: on light grey ground it thins the text's soft edges."""
    v = hsv[..., 2]
    bar = TEXT_V
    if strict:
        bar = max(TEXT_V, min(TEXT_V_MAX, int(np.percentile(v, 90)) + TEXT_V_OVER))
    return (v > bar) & (hsv[..., 1] < TEXT_S)


def _segments(mask, x0, x1, y0, y1):
    """Column runs of text inside mask[y0:y1, x0:x1] as (x_start, x_end, rows_lo, rows_hi)."""
    band = mask[y0:y1, x0:x1]
    cols = band.any(axis=0)
    runs, start = [], None
    for i, c in enumerate(list(cols) + [False]):
        if c and start is None:
            start = i
        elif not c and start is not None:
            rows = np.nonzero(band[:, start:i].any(axis=1))[0]
            runs.append((x0 + start, x0 + i, y0 + rows.min(), y0 + rows.max() + 1))
            start = None
    return runs


def _digit_patch(gray, xs, xe, ly):
    """A fixed-size window centred on the glyph, blank outside its own columns (+1px):
    a narrow "1" would otherwise be half made of whatever stands beside it -- a
    neighbouring digit, a map icon -- and stop looking like the other 1s."""
    cx = (xs + xe) // 2
    x0 = cx - DIGIT_W // 2
    p = _patch(gray, x0, ly + DIGIT_ROWS[0], DIGIT_W, DIGIT_ROWS[1] - DIGIT_ROWS[0] + 15)
    keep = np.zeros(DIGIT_W, bool)
    keep[max(0, xs - 1 - x0):max(0, xe + 1 - x0)] = True
    p[:, ~keep] = 0
    return p


def _cut(gray, hsv, lx, ly, strict=False):
    """The glyph runs after a letter whose top-left is (lx, ly): list of ("." | patch)."""
    mask = _text_mask(hsv, strict)
    top = ly - 4                                  # digits stand 4px taller than the x
    runs = _segments(mask, lx + 9, lx + 9 + 75, top - 1, top + 16)
    out = []
    for xs, xe, r0, r1 in runs:
        w, h = xe - xs, r1 - r0
        if h <= 5 and w <= 4 and r0 >= top + 10:
            out.append((".", xs, xe))
        elif h >= 10 and 2 <= w <= 12:
            out.append(("digit", xs, xe))
        elif h >= 10 and w > 12:                  # two digits touching: split down the middle
            mid = (xs + xe) // 2
            out += [("digit", xs, mid), ("digit", mid, xe)]
        else:
            out.append(("junk", xs, xe))
    return out


class Glyphs:
    def __init__(self, path=GLYPHS):
        d = np.load(path)
        self.letters = {"x": d["letter_x"], "y": d["letter_y"]}
        self.digits, self.labels = d["digits"], [str(c) for c in d["labels"]]

    def classify(self, patch):
        """(char, score, margin over the best DIFFERENT char) for a mask patch."""
        shifted = [patch] + [np.roll(np.roll(patch, dy, 0), dx, 1)
                             for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        scores = [max(_ncc(p, t) for p in shifted) for t in self.digits]
        order = np.argsort(scores)[::-1]
        best = self.labels[order[0]]
        other = next(scores[i] for i in order if self.labels[i] != best)
        return best, scores[order[0]], scores[order[0]] - other

    def find_letter(self, gray, which, ex, ey):
        """Best match of the letter template near (ex, ey): ((lx, ly), score)."""
        t = self.letters[which]
        bx, by, bw, bh = LETTER_BOX
        win = _patch(gray, ex + bx - SEARCH, ey + by - SEARCH, bw + 2 * SEARCH, bh + 2 * SEARCH)
        res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        return (ex - SEARCH + mx, ey - SEARCH + my), float(score)


def read_label(gray, hsv, glyphs, which, cx, cy, strict=False):
    """One label's value as a string like "98.99", or (None, why)."""
    ex, ey = cx + LETTER_AT[which][0], cy + LETTER_AT[which][1]
    (lx, ly), score = glyphs.find_letter(gray, which, ex, ey)
    if score < MIN_LETTER:
        return None, f"no '{which}' label near the cursor (best match {score:.2f})"
    maskf = _text_mask(hsv, strict).astype(np.float32)
    text = ""
    for kind, xs, xe in _cut(gray, hsv, lx, ly, strict):
        if kind == ".":
            if "." in text or not text:
                return None, f"'{which}': a stray point after {text!r}"
            text += "."
        elif kind == "digit":
            ch, s, margin = glyphs.classify(_digit_patch(maskf, xs, xe, ly))
            if s < MIN_SCORE or margin < MIN_MARGIN:
                return None, (f"'{which}': unreadable digit after {text!r} "
                              f"(best {ch} at {s:.2f}, {margin:.2f} clear of the next)")
            text += ch
        elif "." in text and len(text.split(".")[1]) == 2:
            break
        else:
            return None, f"'{which}': something not a digit after {text!r}"
        if "." in text and len(text.split(".")[1]) == 2:
            break
    if not re.fullmatch(r"\d{1,3}\.\d\d", text):
        return None, f"'{which}': read {text!r}, not a coordinate"
    return text, ""


def read(bgr, cursor, origin=(0, 0), glyphs=None):
    """Both labels for a crosshair at frame point `cursor`; `bgr` may be a crop whose
    top-left sits at frame point `origin`."""
    glyphs = glyphs or _default_glyphs()
    gray, hsv = _gray_hsv(bgr)
    cx, cy = cursor[0] - origin[0], cursor[1] - origin[1]
    vals, whys = {}, []
    for which in ("x", "y"):
        v, why = read_label(gray, hsv, glyphs, which, cx, cy)
        if v is None and not why.startswith("no "):     # the letter is there, its digits blurred
            v2, why2 = read_label(gray, hsv, glyphs, which, cx, cy, strict=True)
            if v2 is not None:
                v, why = v2, ""
        vals[which] = float(v) if v else None
        if why:
            whys.append(why)
    return Reading(vals["x"], vals["y"], "; ".join(whys))


_glyphs = None


def _default_glyphs():
    global _glyphs
    if _glyphs is None:
        _glyphs = Glyphs()
    return _glyphs


# --- building glyphs.npz from the labelled captures ------------------------------------
NAME = re.compile(r"x(\d+\.\d\d)_y(\d+\.\d\d)_at(\d+)-(\d+)")


def samples(folder=SAMPLES):
    """(path, {"x": "98.99", "y": "50.77"}, (cx, cy)) for each labelled capture."""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.*"))):
        m = NAME.search(os.path.basename(p))
        if m:
            out.append((p, {"x": m[1], "y": m[2]}, (int(m[3]), int(m[4]))))
    return out


def _letter_topleft(hsv, which, cx, cy):
    """Exact top-left of a letter near its measured offset, from the text mask."""
    ex, ey = cx + LETTER_AT[which][0], cy + LETTER_AT[which][1]
    m = _text_mask(hsv)[ey - 4:ey + 16, ex - 4:ex + 8]
    ys, xs = np.nonzero(m)
    return ex - 4 + int(xs.min()), ey - 4 + int(ys.min())


def build(folder=SAMPLES, out=GLYPHS, skip=()):
    """Cut letter and digit exemplars out of the labelled captures and save them."""
    letters = {"x": [], "y": []}
    digits, labels = [], []
    for path, truth, (cx, cy) in samples(folder):
        if os.path.basename(path) in skip:
            continue
        gray, hsv = _gray_hsv(cv2.imread(path))
        maskf = _text_mask(hsv).astype(np.float32)
        for which in ("x", "y"):
            lx, ly = _letter_topleft(hsv, which, cx, cy)
            bx, by, bw, bh = LETTER_BOX
            letters[which].append(_patch(gray, lx + bx, ly + by, bw, bh))
            cuts = [c for c in _cut(gray, hsv, lx, ly) if c[0] == "digit"]
            want = truth[which].replace(".", "")
            if len(cuts) < len(want):
                raise ValueError(f"{os.path.basename(path)} {which}: cut {len(cuts)} digits, "
                                 f"the label has {len(want)}")
            for (_, xs, xe), ch in zip(cuts, want):
                digits.append(_digit_patch(maskf, xs, xe, ly))
                labels.append(ch)
    np.savez(out, letter_x=np.mean(letters["x"], axis=0).astype(np.float32),
             letter_y=np.mean(letters["y"], axis=0).astype(np.float32),
             digits=np.array(digits, np.float32), labels=np.array(labels))
    return len(digits), sorted(set(labels))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", nargs="?")
    ap.add_argument("--at", help="the crosshair's frame point, x,y")
    ap.add_argument("--build", action="store_true", help=f"rebuild {os.path.basename(GLYPHS)}")
    args = ap.parse_args(argv)
    if args.build:
        n, chars = build()
        print(f"{n} digit exemplars ({''.join(chars)}) -> {GLYPHS}")
        return 0
    if not (args.image and args.at):
        ap.error("give an image and --at x,y (or --build)")
    r = read(cv2.imread(args.image), tuple(int(v) for v in args.at.split(",")))
    print(f"x={r.x} y={r.y}" + (f"   ({r.why})" if r.why else ""))
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
