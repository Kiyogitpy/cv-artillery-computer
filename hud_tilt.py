"""Read the SPH-2's tilt dial -- the tank icon between two tick scales at the bottom
of the gun and driver views -- off a 2560x1440 frame. artillery.py polls it and
corrects the mils for it.

    python hud_tilt.py shot.png [...]      # read whole frames
    python hud_tilt.py --crop crop.png     # read a crop that holds the dial

Measured 2026-09-29 on three screenshots (test_shots/tilt, the USER's):
  - Two scales of seven white ticks, 22.33 px apart, x 1178-1184 and 1377-1383 on the
    frame, the centre tick on row 1261; the icon is centred on x 1280.
  - PITCH: the two triangles outside the scales move up and down together. Nose up
    moves them UP (the USER's uphill shot: 1.55 ticks up). How many degrees a tick is
    has not been measured: DEG_PER_TICK is 4.4 from assuming the uphill and side-on
    shots sat on one even slope (the tilt came out equal in both), which is rough --
    the uphill one was taken moving. A set of shots pivoting on one slope settles it.
  - 2026-09-30 evening, the USER on one flat concrete slope: broadside the roll read 7.0 deg
    with pitch 0; nose up the pitch read 4.3 deg (0.98 ticks at 4.4) with roll 0. One plane
    has one gradient, so both are ~7 deg: a tick is 7/0.98 = ~7.15 deg (DEG_PER_TICK), i.e. the
    old 4.4 was 1.6x too small and every pitch correction since 2026-09-29 was too weak.
    Assumes the bars' angle is the true roll; a 4-heading pivot on that slope checks both
    (pitch = g cos h, roll = g sin h, so hypot(pitch, roll) stays constant).
    MEASURED the same evening with tilt_fit.py (debug/tilt_20260930-175204.csv, 236 samples
    turning the hull on that slope): (pitch, roll) trace an ELLIPSE with its axes on the
    pitch/roll axes (tilt -0.2 deg, so roll lags pitch by exactly 90), rms 0.018 off it;
    pitch centre 0.000 ticks (PITCH_ZERO_PX is right), amplitude 1.001 ticks; roll centre
    +0.2 deg, amplitude 7.26 deg. Ratio 7.25 deg per tick if the bars are true degrees --
    still UNPROVEN in absolute terms (the terrain grid has no ramp there and cannot judge).
    ABSOLUTE CHECK (same evening): a chase-camera screenshot broadside on the slope, HUD
    roll -7.0, showed the tank's two orange tail lights (centres (987,1262) and (1474,1318)
    on the 2560x1440 frame) on a line 6.61 deg down to the RIGHT -- left side up, so the
    sign is right, and 6.6 vs 7.0 is what the camera's downward look (~20 deg, cos = 0.94)
    predicts for a true 7.0. So the bars ARE degrees, and 7.25 deg per tick stands.
    The bearing tape was useless in that run (4 Hz sampling, aliased): the heading is not
    needed for the amplitudes, only for the sign and the uphill direction.
  - ROLL: the two bars inside the scales are ONE line through the centre tick that
    tilts (its midpoint stayed on the centre tick in all three shots, and each bar's
    own slope matched the line through both). Its angle is read as the roll, + when
    the RIGHT end is higher. SIGN CONFIRMED 2026-09-30 (tilt_fit run): the USER began a
    RIGHT turn (heading up) and roll read -7 as the pitch fell through zero, i.e. facing
    90 deg clockwise of uphill, where the LEFT side is the high one -- so a negative roll
    is left side up, right end of the bars up = right side up, as tilt_mil assumes.
  - TURRET: the barrel line on the icon, the only thin straight line radiating from
    the turret's pivot; + clockwise (to the right of the hull's nose). The hull icon
    itself never turns.
  - "Stabilized" draws the triangles and bars white; "Unstabilized" (seen at 8-10
    degrees of roll) draws them orange. WRONG AS A TEST (2026-09-30): the whole hull
    pivot below showed "Unstabilized" in white, and read as stabilized every frame. The
    word itself says which; `stabilized` is not to be trusted until it reads the word.
  - ZERO (2026-09-30): a full hull pivot on flat concrete (1383 frames) read pitch
    +0.134 +- 0.002 ticks at every heading, roll 0.00 -- a slope would swing the pitch
    with the heading, so this is where a LEVEL hull draws the triangles: 3.0 px above the
    centre tick (PITCH_ZERO_PX, subtracted). Uncorrected, it read 0.6 deg nose up and
    moved every SPH-2 mil ~10 on flat ground.

Nothing here guesses: a dial it cannot find whole is a reason, not a reading.
"""
import argparse
import math
import sys
from dataclasses import dataclass

import cv2
import numpy as np

BOX = (1140, 1120, 280, 250)      # x, y, w, h on the frame: the dial with room to spare
DEG_PER_TICK = 7.25               # roll amplitude / pitch amplitude on one slope (tilt_fit), see the docstring
PITCH_ZERO_PX = 3.0               # a LEVEL hull draws the triangles this far above the centre tick

TOPHAT = 9
INK = 60                          # top-hat level of a HUD stroke (the samples: 115-220)
INK_PALE = 45                     # ...tried when no dial shows at INK: over pale concrete the white ticks
                                  # top-hat lower (live 2026-10-14: 190 forward-turret reads found no dial;
                                  # at 45, +39 right reads, none wrong past the turret check, and no dial
                                  # in any of 45 map-open frames)
TICKS = 7
TICK_W = (5, 9)                   # a tick is 7 px wide, 2 tall
COLUMN_MIN = 4                    # ticks of 7 each scale must show on its own
SCALE_GAP = (180, 220)            # the left and right scales are 199 px apart
PIVOT_DY = 3                      # the turret turns about a point 3 px below the centre tick
BARREL_R = (36, 80)               # the barrel runs from the turret's front (~33 px) to ~66 px
BARREL_MIN = 16                   # px of barrel it must see
BARREL_START = 4                  # px: the run must begin this close to BARREL_R[0]
BARREL_THIN = 0.6                 # share of the run that must be a thin line (the samples: 0.8+)
ROLL_COVER = 0.7                  # share of both bar spans the fitted roll line must cover
STABILIZED_MAX_SAT = 80           # white strokes; orange ones are 150+


@dataclass
class Tilt:
    pitch_ticks: float            # + nose up, in ticks of the scale
    roll_deg: float               # + right end of the bars higher
    turret_deg: float             # + clockwise from the hull's nose
    stabilized: bool

    @property
    def pitch_deg(self):
        return self.pitch_ticks * DEG_PER_TICK


def _ink(bgr, level=None):
    """Top-hat of the brightest channel: white and orange strokes both stand out of the
    scene behind them (the dial is drawn over whatever the camera sees)."""
    v = bgr.max(axis=2)
    return cv2.morphologyEx(v, cv2.MORPH_TOPHAT, np.ones((TOPHAT, TOPHAT), np.uint8)) > (INK if level is None else level)


def _scales(ink):
    """The two tick columns: ((left x0, x1), (right x0, x1), centre-tick row, spacing),
    or None. The scales are seven evenly spaced short dashes at two x's 199 px apart,
    on the SAME rows -- so a tick one scale loses (live 2026-09-29: it touched a bright
    scene edge and merged into a taller blob) is fine when the other scale has it. Each
    scale needs COLUMN_MIN of the seven; all but one row need a tick on at least one (live
    2026-09-30 18:30: the bright edge of the bonnet swallowed the same row in BOTH scales and
    the dial read as absent for the whole session at that spot)."""
    n, _, st, cen = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    dashes = [(int(st[i, 0]), int(st[i, 0] + st[i, 2] - 1), float(cen[i][1])) for i in range(1, n)
              if TICK_W[0] <= st[i, 2] <= TICK_W[1] and st[i, 3] <= 3]
    columns = {}
    for x0, x1, y in dashes:
        key = next((k for k in columns if abs(k[0] - x0) <= 2 and abs(k[1] - x1) <= 2), (x0, x1))
        columns.setdefault(key, []).append(y)
    columns = {k: ys for k, ys in columns.items() if len(ys) >= COLUMN_MIN}
    best = None
    for l, lys in columns.items():
        for r, rys in columns.items():
            if SCALE_GAP[0] <= r[0] - l[0] <= SCALE_GAP[1]:
                found = _lattice(lys, rys)
                if found and (best is None or found[2] > best[2][2]):
                    best = (l, r, found)
    if best is None:
        return None
    l, r, (centre, spacing, _) = best
    return l, r, centre, spacing


def _lattice(a, b):
    """(centre row, spacing, ticks matched) of the seven evenly spaced rows that both
    columns' dashes (`a`, `b`) fit, or None. Tried from every pair of dash rows as its
    top and bottom tick; each column must match COLUMN_MIN rows, and together all 7."""
    rows = sorted(set(a) | set(b))
    best = None
    for i, top in enumerate(rows):
        for bottom in rows[i + 1:]:
            spacing = (bottom - top) / (TICKS - 1)
            if not 18 <= spacing <= 28:
                continue
            want = [top + k * spacing for k in range(TICKS)]
            hit_a = [any(abs(y - w) <= 1.5 for y in a) for w in want]
            hit_b = [any(abs(y - w) <= 1.5 for y in b) for w in want]
            if (sum(hit_a) >= COLUMN_MIN and sum(hit_b) >= COLUMN_MIN
                    and sum(p or q for p, q in zip(hit_a, hit_b)) >= TICKS - 1):
                score = sum(hit_a) + sum(hit_b)
                if best is None or score > best[2]:
                    best = (top + (TICKS - 1) / 2 * spacing, spacing, score)
    return best


def _blobs(ink, x0, x1, y0, y1):
    """Connected ink inside [x0, x1) x [y0, y1): [(x, y, w, h, area, cx, cy)] in image coordinates."""
    h, w = ink.shape
    x0, x1, y0, y1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return []
    n, _, st, cen = cv2.connectedComponentsWithStats(ink[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    return [(int(st[i, 0]) + x0, int(st[i, 1]) + y0, int(st[i, 2]), int(st[i, 3]), int(st[i, 4]),
             float(cen[i][0]) + x0, float(cen[i][1]) + y0) for i in range(1, n)]


def _triangle(ink, x0, x1, centre, spacing):
    """Centre row of the pitch triangle in the strip [x0, x1), or None."""
    tris = [b for b in _blobs(ink, x0, x1, int(centre - 5 * spacing), int(centre + 5 * spacing))
            if 10 <= b[3] <= 18 and 4 <= b[2] <= 9]
    if len(tris) != 1:
        return None
    x, y, w, h = tris[0][:4]
    return y + (h - 1) / 2, (x, y, w, h)


def _roll(ink, lx1, rx0, centre):
    """Angle of the roll line in degrees (+ right end higher), or None. The two bars are
    one straight line through the dial's centre, each 21 px long and 20 px in from its
    scale, so the line is FITTED: the angle (and a pixel or two of offset) whose line
    covers most of both bar spans. A scene edge touching one bar -- the uphill shot's
    right bar merged with one below it -- cannot pull a fit that must cover both."""
    h, w = ink.shape
    mid = (lx1 + rx0) / 2
    # The bars sit a fixed distance ALONG the line from the centre (s = 57..77 px either
    # way), so a steeply tilted line puts them nearer the middle in x: sampling fixed x spans
    # covered only 65% of a 26 degree roll (live 2026-09-30, orange) and read it as absent.
    d = (rx0 - lx1) / 2 - 30
    s = np.concatenate([np.arange(-d - 10, -d + 10), np.arange(d - 10, d + 10)])[None, None, :]
    angs = np.radians(np.arange(-45, 45.05, 0.2))[:, None, None]    # seen to 27 deg (orange, 2026-09-30)
    offs = np.arange(-4, 5, 1.0)[None, :, None]
    xs = np.rint(mid + s * np.cos(angs)).astype(int) + 0 * offs.astype(int)
    if xs.min() < 0 or xs.max() >= w:
        return None
    ys = np.rint(centre + offs - s * np.sin(angs)).astype(int)
    inside = (ys >= 0) & (ys < h)
    cover = (ink[np.clip(ys, 0, h - 1), xs] & inside).mean(axis=2)
    if cover.max() < ROLL_COVER:
        return None
    ai, _ = np.nonzero(cover >= cover.max() - 0.03)
    return float(np.degrees(angs[ai, 0, 0]).mean())


def _rays():
    """The barrel search's sample points relative to the pivot, worked out once: every
    1 degree, every half pixel along the ray, and 4 px to either side of it."""
    ang = np.radians(np.arange(0, 360, 1.0))[:, None]
    r = np.arange(BARREL_R[0], BARREL_R[1], 0.5)[None, :]
    ux, uy = np.sin(ang), -np.cos(ang)                 # along the ray (clockwise from up)
    nx, ny = -uy, ux                                   # across it
    return ang, (r * ux, r * uy), (r * ux + 4 * nx, r * uy + 4 * ny), (r * ux - 4 * nx, r * uy - 4 * ny)


_RAYS = _rays()


def _barrel(ink, px, py):
    """Clockwise angle of the barrel from straight up, or None. The barrel is the one
    straight line that STARTS at the turret (the run must begin within BARREL_START px
    of BARREL_R[0]) and is THIN along most of its length (clear 4 px either side). The
    roll bars are thin radial lines too, but begin ~57 px out; the hull's tracks start
    close in but are 10 px wide. Thin is clear on EITHER side: live 2026-10-14 a barrel pointing
    back (150-210 deg) runs beside the turret's outline, clear on one side only, and a short
    outline edge 8-12 deg off was taken instead (7 reads of a 2564-read sweep, none after)."""
    h, w = ink.shape
    grown = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    def at(dx, dy, mask):
        xs, ys = px + dx, py + dy
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        return mask[np.clip(np.rint(ys).astype(int), 0, h - 1), np.clip(np.rint(xs).astype(int), 0, w - 1)] & inside

    ang, (ax, ay), (lx, ly), (rx, ry) = _RAYS
    on = at(ax, ay, grown)
    clear = ~at(lx, ly, ink) | ~at(rx, ry, ink)
    thin = on & clear
    n = on.shape[1]
    idx = np.arange(n)[None, :]
    start = np.where(on.any(axis=1), on.argmax(axis=1), n)[:, None]
    # the run from its first hit, gaps up to 2 px bridged (a closing 5 samples wide)
    closed = cv2.morphologyEx(on.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)) > 0
    past = (idx > start) & ~closed
    end = np.where(past.any(axis=1), past.argmax(axis=1), n)[:, None]
    length = (end - start)[:, 0]
    in_run = (idx >= start) & (idx < end)
    thin_share = (thin & in_run).sum(axis=1) / np.maximum(length, 1)
    best = np.where((start[:, 0] <= 2 * BARREL_START) & (thin_share >= BARREL_THIN), length * 0.5, 0.0)
    if best.max() < BARREL_MIN:
        return None
    good = np.nonzero(best >= best.max() - 2)[0]
    # the angles within 2 px of the longest run form one arc around the barrel (wrap-aware mean)
    a = ang[good, 0]
    if np.degrees(np.ptp(np.unwrap(a))) > 15:
        return None                                    # two lines, or clutter: not one barrel
    return math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def read(bgr, need_turret=True):
    """(Tilt, "") or (None, why) for an image holding the dial (a BOX crop of the frame).
    With need_turret=False an unreadable barrel gives turret_deg = nan instead of no reading
    (a pale barrel on a pale wall is invisible to the top-hat; pitch and roll are still fine,
    and a caller that only wants the ground's slope does not need the turret)."""
    ink = _ink(bgr)
    scales = _scales(ink)
    if scales is None:
        ink = _ink(bgr, INK_PALE)
        scales = _scales(ink)
    if scales is None:
        return None, "no tilt dial (map open, or not in the vehicle?)"
    (lx0, lx1), (rx0, rx1), centre, spacing = scales
    left = _triangle(ink, lx0 - 12, lx0 - 1, centre, spacing)
    right = _triangle(ink, rx1 + 2, rx1 + 13, centre, spacing)
    found = [t for t in (left, right) if t is not None]
    if not found:
        return None, "the pitch triangles are not readable"
    if len(found) == 2 and abs(found[0][0] - found[1][0]) > 2:
        return None, "the two pitch triangles disagree"
    tri_y = sum(t[0] for t in found) / len(found)
    roll = _roll(ink, lx1, rx0, centre)
    if roll is None:
        return None, "the roll bars are not readable"
    turret = _barrel(ink, (lx0 + rx1) / 2, centre + PIVOT_DY)
    if turret is None:
        if need_turret:
            return None, "the barrel on the tank icon is not readable"
        turret = float("nan")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    x, y, w, h = found[0][1]
    sat = hsv[y:y + h, x:x + w, 1][ink[y:y + h, x:x + w]]
    return Tilt((centre - tri_y - PITCH_ZERO_PX) / spacing, roll, turret,
                float(sat.mean()) <= STABILIZED_MAX_SAT), ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--crop", action="store_true", help="the images already hold just the dial")
    args = ap.parse_args(argv)
    x, y, w, h = BOX
    for p in args.images:
        im = cv2.imread(p)
        t, why = read(im if args.crop else im[y:y + h, x:x + w])
        print(f"{p}: " + (why if t is None else
                          f"pitch {t.pitch_ticks:+.2f} ticks ({t.pitch_deg:+.1f} deg), roll {t.roll_deg:+.1f} deg, "
                          f"turret {t.turret_deg:+.1f} deg, {'Stabilized' if t.stabilized else 'Unstabilized'}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
