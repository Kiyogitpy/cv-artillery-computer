"""Firing solutions: a line-for-line port of wardogs-calculator's maths (MIT; see
wardogs/VENDORED.md for the source commit and which JS function each piece mirrors).

Positions are the game's map units (the "x98.99" / "y50.77" the map shows), 100 m
each; +y is north. Azimuth is degrees clockwise from north. Mils come from the
weapon's measured range tables in wardogs/weapons.json: the SPH-2 has a low and a
high arc, the L81 mortar a single one.
"""
import json
import math
import os
from dataclasses import dataclass, field

WEAPONS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wardogs", "weapons.json")
METERS_PER_UNIT = 100   # coordinateMetersPerUnit, the same on all three maps
ARCS = ("single", "low", "high")


@dataclass
class Weapon:
    id: str
    name: str
    min_m: float
    max_m: float
    tables: dict            # arc -> [(distance_m, mil), ...]


@dataclass
class Mil:
    mil: float | None       # None when the table holds several mils at exactly this range
    min_mil: float
    max_mil: float


@dataclass
class Solution:
    distance_m: float
    azimuth_deg: float
    in_range: bool
    mils: dict = field(default_factory=dict)   # arc -> Mil, only the arcs the weapon has
    height: dict = field(default_factory=dict)  # arc -> (height-corrected mil or None, why)


def load_weapons(path=WEAPONS_JSON):
    """{id: Weapon} in file order, as normalizeWeapon/normalizeBallistics read it."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for item in data["weapons"]:
        max_km = float(item.get("maxRangeKm", item.get("rangeKm", item.get("range"))))
        min_km = float(item.get("minRangeKm", 0))
        tables = {}
        for arc in ARCS:
            rows = [(float(d), float(m)) for d, m in (item.get("ballistics") or {}).get(arc, [])]
            if rows:
                tables[arc] = rows
        out[item["id"]] = Weapon(item["id"], item.get("names", {}).get("en", item["id"]),
                                 min_km * 1000, max_km * 1000, tables)
    return out


def _group(table):
    """groupBallisticTable: sorted by (distance, mil), mils at one distance together."""
    groups = []
    for d, mil in sorted(table):
        if groups and groups[-1][0] == d:
            groups[-1][1].append(mil)
        else:
            groups.append((d, [mil]))
    return groups


def _closest(values, target):
    """closestMil: the first value nearest the target."""
    best = values[0]
    for v in values:
        if abs(v - target) < abs(best - target):
            best = v
    return best


def interpolate(table, distance_m):
    """interpolateBallisticTable: a Mil, or None outside the table."""
    if not table or not math.isfinite(distance_m):
        return None
    groups = _group(table)
    for d, mils in groups:
        if abs(d - distance_m) <= 1e-6:
            return Mil(mils[0] if len(mils) == 1 else None, min(mils), max(mils))
    for (ld, lm), (rd, rm) in zip(groups, groups[1:]):
        if ld < distance_m < rd:
            right_avg = sum(rm) / len(rm)
            left = _closest(lm, right_avg)
            right = _closest(rm, left)
            mil = left + (distance_m - ld) / (rd - ld) * (right - left)
            return Mil(mil, mil, mil)
    return None


LOW_HEIGHT_JSON = os.path.join(os.path.dirname(WEAPONS_JSON), "low-main.json")
_low_height = None


def _low_model():
    """(distance nodes, boundary mils, height at each boundary x node) from low-main.json."""
    global _low_height
    if _low_height is None:
        with open(LOW_HEIGHT_JSON, encoding="utf-8") as f:
            r = json.load(f)["representation"]
        lo, hi = r["minDeltaZMetersByBoundary"], r["maxDeltaZMetersByBoundary"]
        mid = [[(a + b) / 2 for a, b in zip(ra, rb)] for ra, rb in zip(lo, hi)]
        _low_height = (r["distanceNodes"], r["boundariesMrad"], mid)
    return _low_height


def low_arc_height_mil(distance_m, dz_m):
    """SPH-2 low-arc mil for a target dz_m ABOVE the gun (negative: below), from the
    calculator's own low-arc model (wardogs/low-main.json): (mil, "") or (None, why).

    The model stores, for each 10-mil step and each distance, the height difference
    at which the right setting crosses that step; this reads it back continuously. At
    dz 0 it gives the flat table's mil to within a few tenths. The calculator ships it
    switched off and calls it tested for +-40 m. Against the user's own logged shots
    (2026-09-29, shots.csv) it put three shots at a ~100 m-high target within ~20 m of
    where they landed; nothing above ~100 m has been checked yet."""
    nodes, mils, heights = _low_model()
    if not nodes[0] <= distance_m <= nodes[-1]:
        return None, f"no height data at {distance_m:.0f} m (the low-arc model covers {nodes[0]}-{nodes[-1]} m)"
    i = min(max(0, _bisect(nodes, distance_m)), len(nodes) - 2)
    f = (distance_m - nodes[i]) / (nodes[i + 1] - nodes[i])
    zs = [h[i] + (h[i + 1] - h[i]) * f for h in heights]
    if not zs[0] <= dz_m <= zs[-1]:
        return None, f"height difference {dz_m:+.0f} m is beyond the low arc at this range"
    for k in range(len(zs) - 1):
        if zs[k] <= dz_m <= zs[k + 1]:
            t = (dz_m - zs[k]) / (zs[k + 1] - zs[k])
            return mils[k] + t * (mils[k + 1] - mils[k]), ""
    return None, "height model lookup failed"


HIGH_HEIGHT_JSON = os.path.join(os.path.dirname(WEAPONS_JSON), "high-v2.json")
_high_height = None


def _high_model():
    """(domain, boundaries) from high-v2.json: boundaries are (mil, segments) ascending."""
    global _high_height
    if _high_height is None:
        with open(HIGH_HEIGHT_JSON, encoding="utf-8") as f:
            p = json.load(f)
        r = p["representation"]
        _high_height = (p["domain"], r["clipMeters"],
                        [(b["boundaryMrad"], b["segments"]) for b in r["boundaries"]])
    return _high_height


def _boundary_dz(segments, boundary_mil, flat_mil, distance_m, clip):
    """Height difference at which the SPH-2's high arc crosses this 10-mil boundary at
    this range (midpoint of the model's envelope): the segment's own curve where it has
    one, else the clip -- above the flat mil for boundaries below it, below for above."""
    for seg in segments:
        nodes = seg["distanceNodes"]
        if nodes[0] <= distance_m <= nodes[-1]:
            i = min(_bisect(nodes, distance_m), len(nodes) - 2)
            d0, d1 = nodes[i], nodes[i + 1]
            f = 0.0 if d1 == d0 else (distance_m - d0) / (d1 - d0)
            lo, hi = seg["minDeltaZMeters"], seg["maxDeltaZMeters"]
            return ((lo[i] + (lo[i + 1] - lo[i]) * f) + (hi[i] + (hi[i + 1] - hi[i]) * f)) / 2
    return clip if boundary_mil < flat_mil else -clip if boundary_mil > flat_mil else 0.0


def high_arc_height_mil(distance_m, dz_m, flat_mil):
    """SPH-2 high-arc mil for a target dz_m ABOVE the gun (negative: below), from the
    calculator's high-v2 command surface (wardogs/high-v2.json): (mil, "") or (None, why).
    At dz 0 it is exactly `flat_mil`.

    Read CONTINUOUSLY like low_arc_height_mil: the file stores, per 10-mil boundary, the
    height difference at which the right setting crosses it; the shell lands further the
    LOWER the mil (a higher target wants a smaller one), so the boundaries' heights fall as
    the mil rises and dz is bracketed between two of them. UPSTREAM CALLS THIS FILE A
    DISABLED RESEARCH CANDIDATE, held-out validation still required, and it has not been
    checked against real impacts here either. `flat_mil` is the flat table's mil at this
    range (which side of the flat each boundary's clip lies on)."""
    domain, clip, boundaries = _high_model()
    if not domain["distanceMinMeters"] <= distance_m <= domain["distanceMaxMeters"]:
        return None, (f"no height data at {distance_m:.0f} m (the high-arc model covers "
                      f"{domain['distanceMinMeters']}-{domain['distanceMaxMeters']} m)")
    if not domain["deltaZMinMeters"] <= dz_m <= domain["deltaZMaxMeters"]:
        return None, f"height difference {dz_m:+.0f} m is beyond the high-arc model (+-{domain['deltaZMaxMeters']:.0f} m)"
    zs = [_boundary_dz(segs, b, flat_mil, distance_m, clip) for b, segs in boundaries]

    def command(dz):
        for k in range(len(zs) - 1):
            if zs[k] >= dz >= zs[k + 1]:
                lo, hi = boundaries[k][0], boundaries[k + 1][0]
                t = 0.0 if zs[k] == zs[k + 1] else (zs[k] - dz) / (zs[k] - zs[k + 1])
                return lo + t * (hi - lo)
        return None

    at, level = command(dz_m), command(0.0)
    if at is None or level is None:
        return None, "height model lookup failed"
    # The surface's own level-ground command sits up to ~3 mil off the flat table (it is
    # built on 10-mil steps), so only the CHANGE for the height is taken from it.
    return flat_mil + (at - level), ""


def _bisect(nodes, x):
    lo, hi = 0, len(nodes) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if nodes[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


def solve(weapon, gun, target, dz_m=None):
    """Solution from gun (x, y) to target (x, y), both in map units. With dz_m (target
    height minus gun height) the SPH-2's low arc also gets a height-corrected mil in
    `height`: {"low": (mil or None, why)}."""
    dx, dy = target[0] - gun[0], target[1] - gun[1]
    distance_m = math.hypot(dx, dy) * METERS_PER_UNIT
    azimuth = math.degrees(math.atan2(dx, dy))
    if azimuth < 0:
        azimuth += 360
    in_range = distance_m + 1e-6 >= weapon.min_m and distance_m <= weapon.max_m + 1e-6
    mils = {arc: interpolate(t, distance_m) for arc, t in weapon.tables.items()} if in_range else {}
    s = Solution(distance_m, azimuth, in_range, mils)
    if dz_m is not None and in_range and "low" in weapon.tables:
        s.height["low"] = low_arc_height_mil(distance_m, dz_m)
    if dz_m is not None and in_range and "high" in weapon.tables:
        flat = mils["high"].mil if mils.get("high") else None
        s.height["high"] = (high_arc_height_mil(distance_m, dz_m, flat) if flat is not None
                            else (None, "the flat table is ambiguous at this range"))
    return s


def adjust(weapon, arc, distance_m, dz_m, long_m, step_m=10.0):
    """Mils to ADD to the last shot's setting to bring it `long_m` metres back (the
    fall-of-shot correction): the arc's mil-per-metre slope at the target times the
    miss. The low arc uses its height model when it covers the shot, else the flat
    table; the high arc always uses the flat table (live 2026-10-13, dz +80 at 1441 m:
    the shots moved 0.273 mil per metre, the flat table says 0.270, while high-v2's
    slope near its +-80 m clip said 0.72 and every F4 overshot 2.7x). None when the
    table has no slope there."""
    def mil(d):
        flat = interpolate(weapon.tables.get(arc), d)
        flat = None if flat is None or flat.mil is None else flat.mil
        if arc == "low" and dz_m is not None:
            m, _ = low_arc_height_mil(d, dz_m)
            if m is not None:
                return m
        return flat

    lo, hi = mil(distance_m - step_m), mil(distance_m + step_m)
    if lo is None or hi is None:
        return None
    return -long_m * (hi - lo) / (2 * step_m)


def barrel(heading_deg, pitch_deg, roll_deg, turret_deg, elev_mil):
    """World (azimuth deg, elevation mil) of a barrel set `turret_deg` clockwise of the hull's
    nose and `elev_mil` above the hull's deck, on a hull facing `heading_deg` and pitched nose
    up by pitch_deg and rolled right-side-up by roll_deg. A rolled hull swings an elevated
    barrel sideways: live 2026-09-30 two low shots from a hull rolled 12.6 deg went 3.3 and
    4.6 deg left of the tape bearing, and this puts them within ~1 deg of where they landed."""
    h, p, r, t = (math.radians(v) for v in (heading_deg, pitch_deg, roll_deg, turret_deg))
    e = elev_mil / 1000
    f = (math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p))
    r0 = (math.cos(h), -math.sin(h), 0.0)
    u0 = (r0[1] * f[2] - r0[2] * f[1], r0[2] * f[0] - r0[0] * f[2], r0[0] * f[1] - r0[1] * f[0])
    right = tuple(a * math.cos(r) + b * math.sin(r) for a, b in zip(r0, u0))
    up = tuple(b * math.cos(r) - a * math.sin(r) for a, b in zip(r0, u0))
    v = tuple(math.cos(e) * math.cos(t) * fa + math.cos(e) * math.sin(t) * ra + math.sin(e) * ua
              for fa, ra, ua in zip(f, right, up))
    return math.degrees(math.atan2(v[0], v[1])) % 360, math.asin(max(-1.0, min(1.0, v[2]))) * 1000


def aim(heading_deg, pitch_deg, roll_deg, az_deg, el_mil, turret0=0.0):
    """(turret_deg, dial_mil) that put the barrel of a hull at this attitude on WORLD azimuth
    az_deg and world elevation el_mil, or None when the tilted gun cannot reach it. The
    tape bearing to traverse to is heading_deg + turret_deg. Newton from turret0 (the
    current turret angle is a good start), numeric Jacobian."""
    t, e = turret0, el_mil
    for _ in range(40):
        a, w = barrel(heading_deg, pitch_deg, roll_deg, t, e)
        da, dw = (az_deg - a + 180) % 360 - 180, el_mil - w
        if abs(da) < 1e-6 and abs(dw) < 1e-4:
            return t, e
        eps = 1e-4
        a1, w1 = barrel(heading_deg, pitch_deg, roll_deg, t + eps, e)
        a2, w2 = barrel(heading_deg, pitch_deg, roll_deg, t, e + eps * 10)
        j11, j21 = ((a1 - a + 180) % 360 - 180) / eps, (w1 - w) / eps
        j12, j22 = ((a2 - a + 180) % 360 - 180) / (eps * 10), (w2 - w) / (eps * 10)
        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-12:
            return None
        t += (da * j22 - dw * j12) / det
        e += (j11 * dw - j21 * da) / det
    return None


def tilt_mil(mil, pitch_deg, roll_deg, turret_deg):
    """Mils to ADD to the dial so a gun on a tilted hull fires at `mil` of real
    elevation -- what the tables give for level ground. None when the tilted gun
    cannot reach that elevation at all.

    The dial's mil is elevation measured from the HULL, so a hull pitched nose up by
    p (+) and rolled right-side up by r (+), with the turret turret_deg clockwise of
    the nose, points the barrel at a world elevation e_w where
        sin e_w = cos e * (cos t sin p + sin t sin r cos p) + sin e * cos r cos p
    for a dialled elevation e. This solves it for e. It is the rotation of
    wardogs-calculator's experimental SPH-2 platform correction (its sphPlatformBasis,
    roll sign flipped so + is right side up), fed the tilt the dial shows instead of
    that file's fixed 1 degree. At low elevation it is about the tilt along the gun
    -- p cos t + r sin t -- but not on the high arc: at 74 deg a 5 deg nose-up hull
    with the gun square to it still lowers the barrel ~14 mil, where that sum says 0.
    Like the upstream model it takes a mil to be a milliradian of elevation."""
    p, r, t = (math.radians(v) for v in (pitch_deg, roll_deg, turret_deg))
    a = math.cos(t) * math.sin(p) + math.sin(t) * math.sin(r) * math.cos(p)
    b = math.cos(r) * math.cos(p)
    s = math.sin(mil / 1000) / math.hypot(a, b)
    if abs(s) > 1:
        return None
    return (math.asin(s) - math.atan2(a, b)) * 1000 - mil


def miss(gun, target, impact):
    """Where a shell landed relative to its target, in metres: (long, right) --
    positive long = beyond the target along the line of fire, positive right = to
    the right of that line as seen from the gun. All points in map units."""
    ux, uy = target[0] - gun[0], target[1] - gun[1]
    n = math.hypot(ux, uy)
    if n == 0:
        return None
    ux, uy = ux / n, uy / n
    dx, dy = impact[0] - target[0], impact[1] - target[1]
    return (dx * ux + dy * uy) * METERS_PER_UNIT, (dx * uy - dy * ux) * METERS_PER_UNIT


def js_round(v):
    """JavaScript's Math.round: halves go UP (Python's round() sends 844.5 to 844)."""
    return math.floor(v + 0.5)


def format_mil(m):
    """formatMilSolution: "845", or "610-620" when the table is ambiguous there."""
    if m is None:
        return None
    lo, hi = js_round(m.min_mil), js_round(m.max_mil)
    if lo != hi:
        return f"{lo}–{hi}"
    return f"{js_round(m.mil) if m.mil is not None else lo}"
