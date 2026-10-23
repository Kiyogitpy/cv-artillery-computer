"""Where the tank is, which way its hull points, and which map it is on -- read off the in-game
minimap by matching it against the maps' terrain exports (maps/<map>_map.png).

    python minimap.py --build                # once: cache every export at the minimap's scale
    python minimap.py --check                # locate every F5 capture, compare with its F2

Measured 2026-10-11/12 (USER captures, test_shots/minimap):
  - The minimap is the bottom-left square of the 2560x1440 screen (MM), north-up, the tank icon
    at its centre, 1.476 m per minimap pixel. It shows in the SPH-2's gun sight too.
  - The exports are terrain LAYER maps (coloured materials, not photos), 16.384 km square in map
    coordinates: left edge map x 0, top edge map y 163.84 (Ozeti: 0.5 m/px, five captures within
    1-2 m of F2). They match the grey minimap on shape, not brightness (features(): a band-pass over
    local contrast): 0.94-0.97 at the right spot on Ozeti/Bakurani. Zestafona's minimap is dark and
    blurry over open ground, so fewer places there are placed at all -- but none wrongly.
  - The icon's long axis, pointed away from its black dot (toward the gun barrel's end), is the
    hull's compass heading: it fitted all 12 Zestafona tilt stations within 2 degrees.

Nothing here guesses: a minimap it cannot place is a reason, not a fix.
"""
import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(HERE, "maps")
CACHE_DIR = os.path.join(MAPS_DIR, "cache")
EXPORTS = {"bakurani": "bakurani_map.png", "ozeti": "ozeti_map.png", "zestafona": "zesty_map.png"}
SPAN_M = 16384.0                # every export spans this many metres of map coordinates
MM = (43, 1007, 341, 306)       # x, y, w, h of the minimap on the 2560x1440 frame
ICON = (170, 153)               # the tank icon's centre inside the minimap crop
MM_M_PER_PX = 1.476             # metres per minimap pixel
COARSE_M_PER_PX = 8.0           # the whole-map search's scale

# A fix needs a match of at least MIN_SCORE and a lead of MIN_LEAD over the best spot outside its
# peak. Measured on the 24 captures searched as tracking would (2026-10-12, features()): right fixes
# on Ozeti/Bakurani scored 0.94-0.97 with a 0.78+ lead; Zestafona's minimap is darker and blurrier
# and shows open ground the export draws as mottled texture, so right fixes there scored 0.17-0.56
# with leads 0.10-0.40 -- and every WRONG fix scored 0.09 or less with a lead of 0.03 or less.
# Featureless ground is a reason, not a fix.
MIN_SCORE = 0.25
MIN_LEAD = 0.15
TRACK_M = 60.0                  # around the last fix, the search reaches this far...
DRIFT_M_PER_S = 15.0            # ...plus this per second since it (the track never expires: the big
MAX_REACH_M = 300.0             #     map open hides the minimap, and Zestafona's whole-map search
                                #     cannot find it again -- review 2026-10-12), up to this
JUMP_M = 60.0                   # a fix this far from the last (or on another map) needs a second...
AGREE_M = 10.0                  # ...read within this of it, within PENDING_S, before it is taken
PENDING_S = 2.0
GLOBAL_EVERY_S = 2.0            # a whole-map search at most this often while tracking fails...
GLOBAL_MAX_S = 16.0             # ...the gap doubling after each that finds nothing, up to this
                                #     (it takes ~0.4 s; lost on Zestafona it never finds anything)
# Our own tank icon, measured on 24 captures: area 523-546 px, long/short side 1.89-1.98, centred
# within 2.2 px of ICON. A blob outside these is not (only) our icon -- a teammate's touching it.
# Live 2026-10-14 a teammate's diamond marker sat ON our icon for a whole session and covered part of
# it: 410-429 px, refused as "not alone" on 83 of 86 crops though the dot still showed. From 380, those
# read within 0.2-0.7 deg (median) of the uncovered reads either side of them.
ICON_AREA = (380, 650)
ICON_ASPECT = (1.7, 2.2)
ICON_OFF_PX = 6.0
DOT_V = 45                      # the icon's dot is near black (its turret-ring edge is ~80-130)
DOT_OFF_PX = 2.0                # ...and this far along the icon from its centre, or no heading


@dataclass
class Fix:
    map: str
    x: float                    # map units (100 m)
    y: float
    heading: float = None       # the hull's compass heading, or None if the icon was unclear
    score: float = 0.0
    lead: float = 0.0


def cache_paths(map_id):
    return (os.path.join(CACHE_DIR, f"{map_id}_fine.npy"), os.path.join(CACHE_DIR, f"{map_id}_coarse.npy"))


def build(map_ids=None, out=CACHE_DIR):
    """Resample each export to the minimap's scale (fine) and COARSE_M_PER_PX (coarse), greyscale,
    as .npy files the window loads in well under a second. Returns {map: (fine shape, coarse shape)}."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(out, exist_ok=True)
    done = {}
    for m in map_ids or EXPORTS:
        src = os.path.join(MAPS_DIR, EXPORTS[m])
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert("L")
        fine_n = int(round(SPAN_M / MM_M_PER_PX))
        coarse_n = int(round(SPAN_M / COARSE_M_PER_PX))
        fine = np.asarray(im.resize((fine_n, fine_n), Image.BOX, reducing_gap=3.0), np.uint8)
        coarse = np.asarray(im.resize((coarse_n, coarse_n), Image.BOX, reducing_gap=3.0), np.uint8)
        del im
        fp, cp = cache_paths(m) if out == CACHE_DIR else (os.path.join(out, f"{m}_fine.npy"),
                                                           os.path.join(out, f"{m}_coarse.npy"))
        np.save(fp, fine)
        np.save(cp, coarse)
        done[m] = (fine.shape, coarse.shape)
    return done


def features(gray):
    """What the grey minimap and the coloured layer export share: a band-pass (difference of
    Gaussians, 2 and 5 px -- tree clumps, tracks, banks) divided by its local strength, so a dark,
    flat minimap (Zestafona's) and a bright, busy export weigh alike. Measured 2026-10-12 on the 24
    captures against the gradient-magnitude edges it replaced: right fixes on Ozeti/Bakurani 0.94-0.97
    (was 0.85-0.93), Zestafona placed 6 of 12 while tracking (was 3) and 4 of 12 from scratch (was 0),
    the best wrong match's lead 0.03 (was 0.06)."""
    g = gray.astype(np.float32)
    d = cv2.GaussianBlur(g, (0, 0), 2.0) - cv2.GaussianBlur(g, (0, 0), 5.0)
    d /= cv2.GaussianBlur(np.abs(d), (0, 0), 12) + 1.0
    return ((d - d.mean()) / (d.std() + 1e-6)).astype(np.float32)


def crop_minimap(frame):
    x, y, w, h = MM
    return frame[y:y + h, x:x + w]


def template(crop):
    """(grey minimap, mask): the icon, the "NAV ..." text and anything tinted (zones, marks) masked."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.where(hsv[..., 1] < 60, 255, 0).astype(np.uint8)
    cv2.circle(mask, ICON, 28, 0, -1)
    cv2.rectangle(mask, (205, 268), (crop.shape[1], crop.shape[0]), 0, -1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), mask


def icon_heading(crop):
    """(the hull's compass heading, "") or (None, why) from the tank icon at the minimap's centre."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    near = np.zeros(crop.shape[:2], np.uint8)
    cv2.circle(near, ICON, 34, 255, -1)
    col = ((hsv[..., 1] > 130) & (hsv[..., 2] > 120)).astype(np.uint8) * 255 & near
    n, lab, stats, cent = cv2.connectedComponentsWithStats(col)
    if n < 2:
        return None, "no tank icon"
    k = 1 + int(np.argmin([math.hypot(*(cent[i] - np.array(ICON))) for i in range(1, n)]))
    if math.hypot(*(cent[k] - np.array(ICON))) > ICON_OFF_PX:
        return None, "no tank icon at the centre"
    if not ICON_AREA[0] <= stats[k, cv2.CC_STAT_AREA] <= ICON_AREA[1]:
        return None, "the tank icon is not alone (a teammate's touching it?)"
    blob = (lab == k).astype(np.uint8)
    cnt = max(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    if not ICON_ASPECT[0] <= max(w, h) / max(min(w, h), 1e-6) <= ICON_ASPECT[1]:
        return None, "the tank icon is not alone (a teammate's touching it?)"
    a = math.radians(ang if w >= h else ang + 90)
    ax, ay = math.cos(a), math.sin(a)
    hull = np.zeros_like(blob)
    cv2.drawContours(hull, [cv2.convexHull(cnt)], -1, 1, -1)
    # The dot: the biggest near-black blob inside the hull shrunk off the icon's dark border. Live
    # 2026-10-12 the mean of every dark pixel in the hull (border, turret-ring edge) sat almost on the
    # centre and the heading flipped 180 degrees from frame to frame on a tank standing still.
    inner = cv2.erode(hull, np.ones((5, 5), np.uint8))
    n2, _, st2, c2 = cv2.connectedComponentsWithStats(((hsv[..., 2] < DOT_V) & (inner > 0)).astype(np.uint8))
    if n2 < 2:
        return None, "no dot on the tank icon"
    j = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
    along = (c2[j][0] - cx) * ax + (c2[j][1] - cy) * ay
    if st2[j, cv2.CC_STAT_AREA] < 3 or abs(along) < DOT_OFF_PX:
        return None, "no dot on the tank icon off its centre (which end is the rear?)"
    if along > 0:                                       # the dot is the rear: face away from it
        ax, ay = -ax, -ay
    return math.degrees(math.atan2(ax, -ay)) % 360, ""


def _peak(r, exclude):
    """(best, best outside `exclude` px of it, sub-pixel (x, y), at the edge) of a match surface.
    A best on the surface's edge may be a cut-off slope of a peak just outside it (review
    2026-10-12: 4 of 2304 windows took one with 11-12 m error)."""
    r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
    _, mx, _, (x, y) = cv2.minMaxLoc(r)
    r2 = r.copy()
    cv2.circle(r2, (x, y), exclude, -1, -1)
    dx = dy = 0.0
    if 0 < x < r.shape[1] - 1:
        a, b, c = r[y, x - 1], r[y, x], r[y, x + 1]
        dx = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0.0
    if 0 < y < r.shape[0] - 1:
        a, b, c = r[y - 1, x], r[y, x], r[y + 1, x]
        dy = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0.0
    edge = x < 2 or y < 2 or x > r.shape[1] - 3 or y > r.shape[0] - 3
    return float(mx), float(r2.max()), (x + dx, y + dy), edge


class Locator:
    """Places minimap crops on the maps. Knows nothing about artillery.

    locate(crop, now) returns (Fix, "") or (None, why). With a recent fix (or a seed: the gun F2
    stored) it searches TRACK_M around it on that map; without one it searches every map's coarse
    copy (at most every GLOBAL_EVERY_S) and refines the best. The first fix, a fix far from the
    last one, or one on another map is only taken once a second read agrees with it."""

    def __init__(self, maps=None, fine_m_per_px=None):
        self.maps = maps                     # {map: (fine uint8, coarse features)} or None: from the cache
        self.fine_m = fine_m_per_px          # metres per fine pixel (None: SPAN_M / fine width)
        self.last = None                     # the last fix taken
        self.last_t = None
        self.pending = None                  # a far fix waiting for a second read, and since when
        self.pending_t = None
        self.next_global = 0.0
        self.global_gap = GLOBAL_EVERY_S
        self.only_map = None                 # the map button's choice: search no other map

    def seed(self, map_id, x, y, now):
        """Track from a known position (F2 on the tank): the search starts there, on that map."""
        if map_id in self.available() and not (self.only_map and map_id != self.only_map):
            self.last, self.last_t, self.pending = Fix(map_id, x, y), now, None
            self.global_gap = GLOBAL_EVERY_S

    # --- maps ---------------------------------------------------------------------
    def available(self):
        if self.maps is None:
            maps = {}                        # all or nothing: a failed load leaves no partial set
            for m in EXPORTS:
                fp, cp = cache_paths(m)
                if os.path.exists(fp) and os.path.exists(cp):
                    maps[m] = (np.load(fp, mmap_mode="r"), features(np.load(cp)))
            self.maps = maps
        return sorted(self.maps)

    def _fine_m(self, m):
        return self.fine_m or SPAN_M / self.maps[m][0].shape[1]

    def _to_map(self, m, u, v):
        f = self._fine_m(m)
        return u * f / 100.0, (SPAN_M - v * f) / 100.0

    def _to_px(self, m, x, y):
        f = self._fine_m(m)
        return x * 100.0 / f, (SPAN_M - y * 100.0) / f

    # --- matching -----------------------------------------------------------------
    def _fine(self, m, gray, mask, u0, v0, reach_px):
        """Search the fine map around icon pixel (u0, v0): (score, lead, icon u, icon v) or None."""
        fine = self.maps[m][0]
        h, w = gray.shape
        x0 = int(round(u0 - ICON[0] - reach_px)) - 4
        y0 = int(round(v0 - ICON[1] - reach_px)) - 4
        x1, y1 = x0 + w + 2 * int(reach_px) + 8, y0 + h + 2 * int(reach_px) + 8
        if x0 < 0 or y0 < 0 or x1 > fine.shape[1] or y1 > fine.shape[0]:
            return None
        win = features(np.asarray(fine[y0:y1, x0:x1]))
        mx, sec, (px, py), edge = _peak(cv2.matchTemplate(win, features(gray), cv2.TM_CCORR_NORMED, mask=mask), 12)
        if edge:
            return None
        return mx, mx - sec, x0 + px + ICON[0], y0 + py + ICON[1]

    def _global(self, gray, mask):
        """Every map's coarse copy: (map, icon u, icon v on the fine map) of the best, or None."""
        best = None
        for m, (fine, coarse_e) in self.maps.items():
            if self.only_map and m != self.only_map:
                continue
            s = MM_M_PER_PX / COARSE_M_PER_PX
            t = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            tm = cv2.resize(mask, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST)
            mx, _, (px, py), _ = _peak(cv2.matchTemplate(coarse_e, features(t), cv2.TM_CCORR_NORMED, mask=tm), 8)
            if best is None or mx > best[0]:
                cm = self._fine_m(m) * fine.shape[1] / coarse_e.shape[1]   # same area as the fine copy
                xm, ym = (px + ICON[0] * s) * cm, (py + ICON[1] * s) * cm
                best = (mx, m, xm / self._fine_m(m), ym / self._fine_m(m))
        return None if best is None else best[1:]

    def locate(self, crop, now):
        if not self.available():
            return None, "no map cache: python minimap.py --build"
        gray, mask = template(crop)
        if mask.mean() < 255 * 0.3:
            return None, "minimap hidden or covered (map open?)"
        if self.only_map:                    # the map button rules: forget anything on another map
            if self.last is not None and self.last.map != self.only_map:
                self.last = None
            if self.pending is not None and self.pending.map != self.only_map:
                self.pending = None
        if self.pending is not None and now - self.pending_t > PENDING_S:
            self.pending = None
        tracking = self.last is not None
        found = None
        # 1. around the last fix (reaching further the longer ago it was), then around a far fix
        #    waiting for its second read
        reach = min(TRACK_M + DRIFT_M_PER_S * (now - self.last_t), MAX_REACH_M) if tracking else TRACK_M
        for near, r_m in ([(self.last, reach)] if tracking else []) + \
                ([(self.pending, TRACK_M)] if self.pending else []):
            r = self._fine(near.map, gray, mask, *self._to_px(near.map, near.x, near.y), r_m / MM_M_PER_PX)
            if r and r[0] >= MIN_SCORE and r[1] >= MIN_LEAD:
                found = (near.map,) + r
                break
        # 2. every map's coarse copy, at most every GLOBAL_EVERY_S
        if found is None and now >= self.next_global:
            self.next_global = now + self.global_gap
            g = self._global(gray, mask)
            if g is not None:
                r = self._fine(g[0], gray, mask, g[1], g[2], 4 * COARSE_M_PER_PX / MM_M_PER_PX)
                if r and r[0] >= MIN_SCORE and r[1] >= MIN_LEAD:
                    found = (g[0],) + r
            self.global_gap = GLOBAL_EVERY_S if found else min(2 * self.global_gap, GLOBAL_MAX_S)
        if found is None:
            return None, "minimap not placed (no confident match: featureless ground?)"
        m, score, lead, u, v = found
        x, y = self._to_map(m, u, v)
        fix = Fix(m, x, y, icon_heading(crop)[0], score, lead)
        far = not tracking or fix.map != self.last.map or \
            math.hypot(fix.x - self.last.x, fix.y - self.last.y) * 100 > JUMP_M
        if far:
            p = self.pending
            if p is None or p.map != fix.map or math.hypot(p.x - fix.x, p.y - fix.y) * 100 > AGREE_M:
                self.pending, self.pending_t = fix, now
                return None, "minimap placed: confirming with a second read"
        self.pending = None
        self.last, self.last_t = fix, now
        self.global_gap = GLOBAL_EVERY_S
        return fix, ""


def check(log=os.path.join(HERE, "test_shots", "minimap", "log.csv")):
    """[(row, Fix or None, why)] for every F5 capture, each located from scratch."""
    out = []
    with open(log, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    loc = Locator()
    for r in rows:
        frame = cv2.imread(os.path.join(os.path.dirname(log), r["file"]))
        loc.last = loc.pending = None
        loc.next_global = 0.0
        crop = crop_minimap(frame)
        fix, why = loc.locate(crop, 0.0)              # the first fix needs a second read
        if fix is None:
            loc.next_global = 0.0
            fix, why = loc.locate(crop, 0.25)
        out.append((r, fix, why))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", nargs="*", metavar="MAP", help="cache these exports (default: all)")
    ap.add_argument("--check", action="store_true", help="locate every F5 capture against its F2")
    args = ap.parse_args(argv)
    if args.build is not None:
        t = time.time()
        for m, shapes in build(args.build or None).items():
            print(f"{m}: fine {shapes[0]}, coarse {shapes[1]}")
        print(f"built in {time.time() - t:.0f} s -> {CACHE_DIR}")
    if args.check:
        for r, fix, why in check():
            if fix is None:
                print(f"{r['file']}: {why}")
                continue
            de, dn = (float(r["gun_x"]) - fix.x) * 100, (float(r["gun_y"]) - fix.y) * 100
            print(f"{r['file']} [{r['map']}] -> {fix.map} x{fix.x:.2f} y{fix.y:.2f} heading "
                  f"{fix.heading if fix.heading is None else round(fix.heading, 1)} score {fix.score:.2f} "
                  f"lead {fix.lead:.2f} | F2 - minimap: east {de:+.1f} m, north {dn:+.1f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
