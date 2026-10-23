"""minimap.Locator on a synthetic map, and on the USER's F5 captures (test_shots/minimap) when the
map caches are built (python minimap.py --build). Run: python -m pytest"""
import csv
import os

import cv2
import numpy as np
import pytest

import minimap as mm

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "test_shots", "minimap")



def read_rows(path):
    """Every row of a CSV file, the file closed again."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def synthetic_map(n=1400, seed=3):
    """A textured 'export': blobs and lines at several scales, so every patch is distinct."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 128, np.uint8)
    for _ in range(900):
        x, y, r = rng.integers(0, n), rng.integers(0, n), int(rng.integers(3, 18))
        cv2.circle(img, (int(x), int(y)), r, int(rng.integers(30, 230)), -1)
    for _ in range(60):
        p = rng.integers(0, n, 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), int(rng.integers(20, 240)), 3)
    return img


@pytest.fixture
def loc():
    fine = synthetic_map()
    # the coarse copy at COARSE_M_PER_PX of a map whose fine pixel is MM_M_PER_PX
    s = mm.MM_M_PER_PX / mm.COARSE_M_PER_PX
    coarse = cv2.resize(fine, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m_per_fine = mm.MM_M_PER_PX
    L = mm.Locator(maps={"syn": (fine, mm.features(coarse))}, fine_m_per_px=m_per_fine)
    return L, fine


def crop_at(fine, u, v):
    """The minimap a tank at fine pixel (u, v) would see: BGR, icon at ICON."""
    x0, y0 = int(u) - mm.ICON[0], int(v) - mm.ICON[1]
    g = fine[y0:y0 + mm.MM[3], x0:x0 + mm.MM[2]]
    return cv2.cvtColor(np.ascontiguousarray(g), cv2.COLOR_GRAY2BGR)


def test_a_seeded_track_reads_the_exact_position(loc):
    L, fine = loc
    x, y = L._to_map("syn", 600, 700)
    L.seed("syn", x + 0.3, y - 0.2, 0.0)                   # F2 ~36 m off
    fix, why = L.locate(crop_at(fine, 600, 700), 0.25)
    assert fix is not None, why
    assert abs(fix.x - x) * 100 < 2 and abs(fix.y - y) * 100 < 2


def test_the_first_fix_from_scratch_needs_a_second_read(loc):
    L, fine = loc
    c = crop_at(fine, 500, 500)
    fix, why = L.locate(c, 0.0)
    assert fix is None and "confirming" in why
    L.next_global = 0.0
    fix, why = L.locate(c, 0.25)
    assert fix is not None, why
    x, y = L._to_map("syn", 500, 500)
    assert abs(fix.x - x) * 100 < 3 and abs(fix.y - y) * 100 < 3


def test_a_far_jump_needs_a_second_read(loc):
    L, fine = loc
    x, y = L._to_map("syn", 500, 500)
    L.seed("syn", x, y, 0.0)
    assert L.locate(crop_at(fine, 500, 500), 0.25)[0] is not None
    far = crop_at(fine, 900, 900)                          # ~800 m away in one read
    L.next_global = 0.0
    fix, why = L.locate(far, 0.5)
    assert fix is None and "confirming" in why
    fix, why = L.locate(far, 0.75)
    assert fix is not None
    fx, fy = L._to_map("syn", 900, 900)
    assert abs(fix.x - fx) * 100 < 3


def test_a_blank_or_hidden_minimap_is_a_reason(loc):
    L, fine = loc
    blank = np.full((mm.MM[3], mm.MM[2], 3), 90, np.uint8)
    fix, why = L.locate(blank, 0.0)
    assert fix is None and why
    tinted = np.zeros((mm.MM[3], mm.MM[2], 3), np.uint8)
    tinted[...] = (200, 40, 40)                            # all saturated: the map screen, a menu
    fix, why = L.locate(tinted, 0.0)
    assert fix is None and "hidden" in why


def test_no_cache_is_a_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "CACHE_DIR", str(tmp_path))
    L = mm.Locator()
    fix, why = L.locate(np.zeros((mm.MM[3], mm.MM[2], 3), np.uint8), 0.0)
    assert fix is None and "--build" in why


# --- the USER's captures (2026-10-11/12) ------------------------------------------------------
CACHED = all(os.path.exists(p) for m in mm.EXPORTS for p in mm.cache_paths(m))
ROWS = read_rows(os.path.join(SHOTS, "log.csv")) \
    if os.path.exists(os.path.join(SHOTS, "log.csv")) else []
real = pytest.mark.skipif(not (CACHED and ROWS), reason="map caches or F5 captures missing")


@real
def test_tracking_never_gives_a_wrong_position_on_the_captures():
    import collections
    L = mm.Locator()
    placed = collections.Counter()
    for r in ROWS:
        frame = cv2.imread(os.path.join(SHOTS, r["file"]))
        gx, gy = float(r["gun_x"]), float(r["gun_y"])
        L.seed(r["map"], gx + 0.3, gy - 0.3, 0.0)           # start ~42 m off, as after a short drive
        fix, why = L.locate(mm.crop_minimap(frame), 0.25)
        if fix is None:
            continue
        assert fix.map == r["map"]
        assert np.hypot(fix.x - gx, fix.y - gy) * 100 < 5, (r["file"], fix)
        placed[r["map"]] += 1
    # Zestafona's minimap is dark and blurry and much of it open ground: fewer confident fixes
    # there, but never a wrong one (2026-10-12, features(): 6 of 12)
    assert placed["ozeti"] >= 5 and placed["zestafona"] >= 6 and placed["bakurani"] >= 5


@real
def test_ozeti_is_found_from_scratch_with_its_map_and_heading():
    L = mm.Locator()
    r = next(r for r in ROWS if r["map"] == "ozeti")
    c = mm.crop_minimap(cv2.imread(os.path.join(SHOTS, r["file"])))
    assert L.locate(c, 0.0)[0] is None                     # first read: confirming
    L.next_global = 0.0
    fix, why = L.locate(c, 0.25)
    assert fix is not None, why
    assert fix.map == "ozeti"
    assert np.hypot(fix.x - float(r["gun_x"]), fix.y - float(r["gun_y"])) * 100 < 3
    assert fix.heading == pytest.approx(243.4, abs=2.0)    # the barrel lower-left in that capture


def test_hidden_reads_and_time_do_not_drop_the_track(loc):
    # review 2026-10-12: the track expired while the big map was open, and on Zestafona the
    # whole-map search never brings it back -- so a closed map must resume tracking
    L, fine = loc
    x, y = L._to_map("syn", 600, 700)
    L.seed("syn", x, y, 0.0)
    hidden = np.zeros((mm.MM[3], mm.MM[2], 3), np.uint8)
    hidden[...] = (200, 40, 40)
    for i in range(40):                                   # 10 s with the big map open
        assert L.locate(hidden, 0.25 * (i + 1))[0] is None
    L.next_global = 1e9                                   # no whole-map search to fall back on
    c = crop_at(fine, 640, 690)                           # it drove ~60 m meanwhile
    fix, why = L.locate(c, 10.5)
    if fix is None:                                       # a move past JUMP_M: one confirming read
        assert "confirming" in why
        fix, why = L.locate(c, 10.75)
    assert fix is not None, why
    x2, y2 = L._to_map("syn", 640, 690)
    assert abs(fix.x - x2) * 100 < 3 and abs(fix.y - y2) * 100 < 3


def test_a_locked_map_is_the_only_one_searched():
    a, b = synthetic_map(seed=3), synthetic_map(seed=8)
    s = mm.MM_M_PER_PX / mm.COARSE_M_PER_PX
    maps = {m: (img, mm.features(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)))
            for m, img in (("a", a), ("b", b))}
    L = mm.Locator(maps=maps, fine_m_per_px=mm.MM_M_PER_PX)
    L.only_map = "a"
    c = crop_at(b, 600, 600)
    for t in (0.0, 0.25):
        L.next_global = 0.0
        fix, _ = L.locate(c, t)
        assert fix is None or fix.map == "a"


def _icon_crop(extra=False):
    """A plain minimap with a red tank icon at ICON (area ~530 px, aspect ~1.95, dot at the rear)."""
    c = np.full((mm.MM[3], mm.MM[2], 3), 70, np.uint8)
    box = cv2.boxPoints(((float(mm.ICON[0]), float(mm.ICON[1])), (32.0, 16.5), 30.0)).astype(np.int32)
    cv2.fillPoly(c, [box], (40, 60, 240))
    a = np.radians(30.0)
    dot = (int(mm.ICON[0] + 7 * np.cos(a)), int(mm.ICON[1] + 7 * np.sin(a)))   # toward the box's +x end
    cv2.circle(c, dot, 3, (10, 10, 10), -1)
    if extra:                                             # a teammate's icon touching ours
        box2 = cv2.boxPoints(((float(mm.ICON[0] + 20), float(mm.ICON[1] - 14)), (32.0, 16.5), 80.0)).astype(np.int32)
        cv2.fillPoly(c, [box2], (40, 60, 240))
    return c


def test_the_icon_heading_points_away_from_the_dot():
    hd, why = mm.icon_heading(_icon_crop())
    assert hd is not None, why
    assert hd == pytest.approx((90 + 30 + 180) % 360, abs=3)   # the box's -x end, compass


def test_a_teammate_touching_the_icon_gives_no_heading():
    # review 2026-10-12: the largest saturated blob near the centre was taken as the hull
    hd, why = mm.icon_heading(_icon_crop(extra=True))
    assert hd is None and why


def test_a_peak_on_the_search_windows_edge_is_refused():
    r = np.zeros((40, 40), np.float32)
    r[20, 39] = 1.0                                       # the best match where the window ends
    assert mm._peak(r, 6)[3] is True
    r2 = np.zeros((40, 40), np.float32)
    r2[20, 20] = 1.0
    assert mm._peak(r2, 6)[3] is False


@real
def test_check_places_the_captures():
    import collections
    placed, wrong = collections.Counter(), 0
    for r, fix, why in mm.check():
        if fix is None or float(r["gun_age_s"] or 0) > 300:
            continue                    # F2's position 2.7 h old (F2 off since 2026-10-12): no truth
        if fix.map == r["map"] and np.hypot(fix.x - float(r["gun_x"]), fix.y - float(r["gun_y"])) * 100 < 5:
            placed[r["map"]] += 1
        else:
            wrong += 1
    assert wrong == 0
    assert placed["ozeti"] >= 5 and placed["bakurani"] >= 7 and placed["zestafona"] >= 3


def test_the_whole_map_search_backs_off_while_nothing_is_found(loc):
    # review 2026-10-12: a lost minimap (always, on Zestafona) re-ran a 0.4 s search every 2 s
    L, fine = loc
    blank = np.full((mm.MM[3], mm.MM[2], 3), 90, np.uint8)
    blank[::7, ::5] = 200                                  # visible, but matches nothing
    gaps, t = [], 0.0
    for _ in range(6):
        L.locate(blank, t)
        gaps.append(L.next_global - t)
        t = L.next_global
    assert gaps[:4] == [2.0, 4.0, 8.0, 16.0] and gaps[4] == 16.0
    x, y = L._to_map("syn", 600, 700)
    L.seed("syn", x, y, t)                                 # F2 (or any fix) starts it over
    L.locate(blank, t + 0.1)
    assert L.next_global - (t + 0.1) <= mm.GLOBAL_EVERY_S + 1e-9


def test_the_same_real_icon_gives_the_same_heading_every_time():
    # live 2026-10-12 (Ozeti, standing still): this icon read 98.7 on one frame and 278.7 on the next
    # -- the "dot" was the mean of every dark pixel in the icon's hull, its dark border included, and
    # sat almost on the centre. A wrong-way trusted prior kept the bearing tape settling for good.
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_shots", "minimap_icon")
    got = [mm.icon_heading(cv2.imread(os.path.join(d, f)))[0] for f in ("ozeti_still_a.png", "ozeti_still_b.png")]
    assert None not in got and abs(got[0] - got[1]) < 2
    assert abs((got[0] - 278.7 + 180) % 360 - 180) < 4        # the dot is the rear: it faces west
