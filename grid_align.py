"""Find how far the interactive map's coordinates sit from the terrain grid's, from tilt readings.

Each STATION is: the hull parked somewhere, F2 pressed with the cursor on the tank ICON
(the icon is the game's own idea of where the tank is; artillery_state.json keeps it), then
the map closed and the tilt dial read with the hull's compass heading. The dial says how the
ground under the tank slopes (pitch and roll); the grid says how it slopes at any position,
so the position the readings fit best is where the tank really was. One station on a smooth
face fits a long band of positions; stations on VARIED ground (crests, gullies, the edge of
a flat pad, corners) each cut a different direction, and the offset that fits all of them
is the answer.

    1. park; map open; F2 (artillery on) with the cursor on the tank icon's centre
    2. close the map, sit in the gun seat (turret forward); DON'T MOVE
    3. python grid_align.py station --heading 162          (the hull's compass bearing)
    ... repeat at 8-10 different spots ...
    python grid_align.py fit                               (the offset and how sure it is)

The offset is metres to ADD to a map position to get the grid position (east, north).
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time

import numpy as np

import terrain

HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = os.path.join(HERE, "alignment_stations.csv")
STATE = os.path.join(HERE, "artillery_state.json")
FIELDS = ["time", "map", "x", "y", "heading", "pitch_deg", "roll_deg", "turret_deg", "note", "gun_time"]
STEP = 2.0             # metres between samples: the source's own resolution
SMOOTH = 3             # samples (6 m) the slope is averaged over: about the hull's footprint
RADIUS_M = 150         # how far from the map position the offset may be
SIGMA_DEG = 1.5        # noise in a pitch or roll reading against the grid (heading, footprint, grid)
OUTLIER_DEG = 5.0      # a station further than this from the grid at the best offset counts as an outlier


def slopes(grid, x, y, half_m=RADIUS_M + 12, step=STEP):
    """East and north slopes (metres per metre) of the smoothed terrain in a window of
    +-half_m around map position (x, y): (dzdx, dzdy) arrays [row = south to north, col =
    west to east], each the window's size less its margin."""
    n = int(round(2 * half_m / step)) + 1
    z = np.empty((n, n))
    for j in range(n):
        for i in range(n):
            v = grid.elevation(x + (i * step - half_m) / 100, y + (j * step - half_m) / 100)
            z[j, i] = np.nan if v is None else v
    if np.isnan(z).any():
        z = np.where(np.isnan(z), np.nanmean(z), z)
    k = np.ones(SMOOTH) / SMOOTH
    z = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, z)
    z = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, z)
    dzdy, dzdx = np.gradient(z, step)
    m = int(round(12 / step))
    return dzdx[m:-m, m:-m], dzdy[m:-m, m:-m]


def predicted(dzdx, dzdy, heading):
    """(pitch_deg, roll_deg) a hull facing `heading` reads on that slope: + nose up, +
    right side up. Same signs as hud_tilt."""
    h = math.radians(heading)
    forward = dzdx * math.sin(h) + dzdy * math.cos(h)
    right = dzdx * math.cos(h) - dzdy * math.sin(h)
    return np.degrees(np.arctan(forward)), np.degrees(np.arctan(right))


def cost_map(station, grid):
    """Squared reading error for every offset in +-RADIUS_M, indexed [north, east]."""
    dzdx, dzdy = slopes(grid, station["x"], station["y"])
    pitch, roll = predicted(dzdx, dzdy, station["heading"])
    return (pitch - station["pitch_deg"]) ** 2 + (roll - station["roll_deg"]) ** 2


def fit(stations, grid_for=None):
    """{'east', 'north', 'chi2', 'east_range', 'north_range', 'per_station'} for the offset
    that best fits every station, with the range of offsets within one standard error.
    Stations are grouped by map; all must share one."""
    grid_for = grid_for or terrain.load
    maps = {s["map"] for s in stations}
    if len(maps) != 1:
        raise ValueError(f"the stations are from more than one map: {sorted(maps)}")
    grid = grid_for(maps.pop())
    maps_ = [cost_map(s, grid) for s in stations]
    # A station on something the grid does not have (a road embankment, a wall) fits NO offset
    # well; its error is capped, so one such reading cannot drag the offset the others agree on.
    total = sum(np.minimum(m, OUTLIER_DEG ** 2) for m in maps_)
    j, i = np.unravel_index(np.argmin(total), total.shape)
    chi2 = total / SIGMA_DEG ** 2
    ok = chi2 <= chi2.min() + 2.3            # ~68% for two parameters
    js, is_ = np.nonzero(ok)
    n = total.shape[0]
    c = (n - 1) / 2
    east = lambda idx: (idx - c) * STEP
    return {"east": east(i), "north": east(j), "chi2": float(chi2.min()), "n": len(stations),
            "east_range": (east(is_.min()), east(is_.max())),
            "north_range": (east(js.min()), east(js.max())),
            "per_station": [float(math.sqrt(m[j, i])) for m in maps_],
            "at_zero": [float(math.sqrt(m[int(c), int(c)])) for m in maps_]}


def load_stations(path=STATIONS):
    with open(path, newline="") as f:
        return [{"x": float(r["x"]), "y": float(r["y"]), "heading": float(r["heading"]),
                 "pitch_deg": float(r["pitch_deg"]), "roll_deg": float(r["roll_deg"]), "map": r["map"]}
                for r in csv.DictReader(f)]


def report(res):
    lines = [f"{res['n']} stations: offset EAST {res['east']:+.0f} m, NORTH {res['north']:+.0f} m "
             f"(map position + this = grid position)",
             f"  within one standard error: east {res['east_range'][0]:+.0f}..{res['east_range'][1]:+.0f}, "
             f"north {res['north_range'][0]:+.0f}..{res['north_range'][1]:+.0f}",
             f"  chi2 {res['chi2']:.1f} for {2 * res['n'] - 2} degrees of freedom "
             f"(about that many = the readings agree with the grid to {SIGMA_DEG} deg)",
             "  per station, tilt error deg: at the offset / with no offset:"]
    for k, (a, b) in enumerate(zip(res["per_station"], res["at_zero"]), 1):
        lines.append(f"    #{k}: {a:.1f} / {b:.1f}" + ("   <- OUTLIER, not counted in the offset" if a > OUTLIER_DEG else ""))
    return "\n".join(lines)


def read_station(heading, note=""):
    import artillery
    from winkeys import dpi_aware
    dpi_aware()
    with open(STATE, encoding="utf-8") as f:
        st = json.load(f)
    if not st.get("gun") or not st.get("map"):
        raise SystemExit("no gun stored or no map picked: F2 on the tank icon and pick the map first")
    age = time.time() - st.get("gun_time", 0)
    if os.path.exists(STATIONS):
        with open(STATIONS, newline="") as f:
            if any(r.get("gun_time") == str(st.get("gun_time")) for r in csv.DictReader(f)):
                raise SystemExit("this F2 position is already a station: press F2 on the tank icon "
                                 "at THIS spot first (a second reading with the old position is a wrong "
                                 "one -- the fit would put the tank where it no longer is)")
    reader = artillery.Reader()
    reads, whys = [], []
    for _ in range(30):                      # a read sometimes fails (a bright scene flickers a bar out)
        t, why = reader.read_tilt(need_turret=False)     # the ground's slope needs no turret
        if t is None:
            whys.append(why)
        else:
            reads.append(t)
        if len(reads) >= 6:
            break
        time.sleep(0.2)
    if len(reads) < 6:
        raise SystemExit(f"tilt readable only {len(reads)} of {len(reads) + len(whys)} times: "
                         f"{max(set(whys), key=whys.count) if whys else '?'}")
    pitch = [t.pitch_deg for t in reads]
    roll = [t.roll_deg for t in reads]
    if max(pitch) - min(pitch) > 0.8 or max(roll) - min(roll) > 0.8:
        raise SystemExit(f"the hull is moving (pitch {min(pitch):.1f}..{max(pitch):.1f}, "
                         f"roll {min(roll):.1f}..{max(roll):.1f}): hold still")
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "map": st["map"], "x": st["gun"][0], "y": st["gun"][1],
           "heading": heading, "pitch_deg": round(statistics.median(pitch), 2),
           "roll_deg": round(statistics.median(roll), 2),
           "turret_deg": (round(statistics.median(turrets), 1) if (turrets := [t.turret_deg for t in reads
                          if t.turret_deg == t.turret_deg]) else ""), "note": note,
           "gun_time": st.get("gun_time", "")}
    new = not os.path.exists(STATIONS)
    with open(STATIONS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"station saved: {row}")
    if age > 600:
        print(f"WARNING: the gun position was stored {age:.0f} s ago -- press F2 on the icon at each station")
    if row["turret_deg"] != "" and abs(row["turret_deg"]) > 3:
        print(f"NOTE: turret at {row['turret_deg']} deg: the heading must be the HULL's")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("station", help="record the tilt at the position F2 last stored")
    s.add_argument("--heading", type=float, required=True, help="the hull's compass bearing, degrees")
    s.add_argument("--note", default="")
    f = sub.add_parser("fit", help="solve for the offset from the saved stations")
    f.add_argument("--csv", default=STATIONS)
    args = ap.parse_args(argv)
    if args.cmd == "station":
        read_station(args.heading % 360, args.note)
        return 0
    stations = load_stations(args.csv)
    if len(stations) < 3:
        print(f"{len(stations)} station(s): need at least 3, and more like 8 on varied ground")
        return 1
    print(report(fit(stations)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
