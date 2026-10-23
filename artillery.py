"""Artillery firing solutions from the in-game map.

Open the map, put the cursor on your gun and press F2: the map's own cursor
readout ("x98.99 y50.77", map_coords.py reads it) becomes the gun position. Then hold F3
with the cursor anywhere on the map: the window shows distance, azimuth and mils to
that point, live, and keeps the last one when you let go. The gun is always the SPH-2.

HEIGHTS COME FROM THE TERRAIN GRID (terrain.py, 2 m), not from any ASL: pick the map with
the window's button (it is remembered) and the height difference between the gun and the
target is read off the grid for both spots -- nothing to type, nothing to read from the
bottom of the screen (that path, and its fitted 0.5 gun-height factor, were removed
2026-09-30). The SPH-2's low arc corrects for it with the calculator's low-arc model
(wardogs/low-main.json) and its high arc with high-v2 (wardogs/high-v2.json; upstream calls it
a research candidate, still to be checked against real impacts -- shots.csv logs `map` and
`dz_m` with every F4 for exactly that). With no map picked the mils assume level ground and
the window says so. The grid itself has NOT been checked against real impacts either
(terrain.py). The maths is wardogs-calculator's (ballistics.py, wardogs/VENDORED.md).

It only reads the screen and shows numbers: nothing is typed or clicked.

    F2          store the gun position (cursor on your gun, on the map)
    F3          HOLD for the solution to the cursor
    F4          where the shell LANDED (cursor on the impact, on the map)
    F5          map CLOSED: save the screen (minimap) with the position F2 stored
    close the window to quit

F4 ADJUSTS FIRE: it reads where the shell landed, shows how far it was long or
short along the line of fire and left or right of it, and from then on the window
shows the mils and azimuth for the next shot corrected by the miss -- the arc's
mil-per-metre slope times the miss, and the angle the miss subtends. The correction
belongs to the GUN, not the target: it applies to every target until F2 stores the gun
again, and each further F4 is averaged into it; it shows as "(adjusted +8)". (Live
2026-09-30, one gun spot on Bakurani: the first shot at each of six new targets up to a
kilometre apart landed 31-65 m long, mean 43 -- the gun's own error, which a per-target
correction threw away at every new target.) It works however the mil came to be off. Each F4 also appends a row to shots.csv with the solution
it was fired on -- distance, azimuth, the mils of every
arc, both heights, and how long ago the gun was stored (a gun that moved since F2
misses by however far it moved), and the tilt dial's last reading. Which arc and
mil you actually fired are not visible to it: the notes column is there to add them.

THE TILT DIAL (hud_tilt.py, 2026-09-29). The SPH-2's mil is elevation measured from
its HULL, so a hull parked on a slope fires high or low: 1 degree is 17.5 mil, ~80 m
of range on the low arc at 1300-1700 m. The dial under "Stabilized" shows the hull's
pitch (triangles), roll (bars) and where the turret points (the barrel on the tank
icon). The window reads it every TILT_EVERY_S while it is on screen -- the gun or
driver view with the map closed -- keeps the last good reading, shows it on the
"Tilt" line, and sets every SPH-2 mil for it ("tilt -12" beside the mil). The
correction depends on where the turret points, so traverse onto the azimuth first
and read the mil after. --no-tilt turns it off. Two things are PROVISIONAL until
measured: how many degrees one pitch tick is (hud_tilt.DEG_PER_TICK, 4.4, rough) and
which way the roll bars lean for a raised right side; the tilt columns in shots.csv
are there to check both against real impacts.

THE MINIMAP (minimap.py, 2026-10-12) is read on the same beat, matched against the maps' terrain
exports on a worker thread: where it places the tank confidently, that IS the gun's position
("Gun ... (minimap)") and the map ("Map: ... (minimap)"), and the tank icon's rotation is the
hull's heading, the bearing tape's trusted prior. A move of more than GUN_MOVED_M is a moved gun
(the F4 correction and the hull anchor go, as with F2); a metre or two of jitter is not. Where it
cannot place it (featureless ground, the big map open) the last position stays and F2 still sets
it by hand -- and seeds the minimap's search. Needs `python minimap.py --build` once.

THE BLUE BEARING LINE ("Bearing +3.2" = traverse 3.2 deg right) reads the bearing tape through
hud_bearing.StepTracker: after the map was open or the gun first comes up it says "Bearing
settling..." until the tape's 15-degree step is confirmed, then follows it. --record-bearing
saves every read for `python hud_bearing.py --replay <folder>`, and every MIL scale read (the
blue Dial line) to <folder>/mil for `python hud_mil.py --strip <file>`.

The gun position, weapon and the window's position are remembered in artillery_state.json
across runs. The window reopens where it was last closed, or the first time on the screen
that is not the game's (--window-x/-y move it); an exclusive-fullscreen game hides anything
drawn over it, which is why.

When the map readout cannot be read -- on F2, or for 0.5 s of holding F3 -- it beeps
low and saves the screen to debug\\ as map_<time>_at<x>-<y>.png (the cursor's
point) with a .txt saying why. Renamed to x<real x>_y<real y>_at<x>-<y>.png and
put in test_shots\\map, `python map_coords.py --build` learns from it.

    python artillery.py
"""
import argparse
import concurrent.futures
import csv
import dataclasses
import json
import math
import os
import sys
import threading
import time
import tkinter as tk

import cv2
import numpy as np

import ballistics
import hud_bearing
import hud_compass
import hud_mil
import hud_tilt
import minimap
import map_coords
import terrain
from winkeys import DEBUG_DIR, Key, beep, cursor, dpi_aware, primary_monitor, side_monitor, vk_code

STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artillery_state.json")
SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots.csv")
# F5: the game screen (minimap showing) with the gun F2 stored, for matching the minimap on the
# terrain map (maps/<map>_map.png). The minimap is hidden while the big map is open.
MINIMAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_shots", "minimap")
MINIMAP_FIELDS = ["time", "map", "gun_x", "gun_y", "gun_age_s", "tilt_pitch_deg", "tilt_roll_deg",
                  "tilt_turret_deg", "bearing_deg", "hull_heading_deg", "file",
                  "gun_source", "mm_x", "mm_y", "mm_heading"]   # gun_x/y/age: F2's own (the truth)
SHOT_FIELDS = ["time", "weapon", "gun_x", "gun_y", "gun_asl", "gun_height_used", "gun_age_s", "target_x", "target_y",
               "target_asl", "dz_m", "distance_m", "azimuth_deg", "mil_single", "mil_low", "mil_high",
               "mil_low_height", "shown_low", "shown_high", "shown_single", "shown_az",
               "next_low", "next_high", "next_single", "next_az",
               "impact_x", "impact_y", "long_m", "right_m", "impact_distance_m", "impact_azimuth_deg",
               "tilt_pitch_ticks", "tilt_pitch_deg", "tilt_roll_deg", "tilt_turret_deg", "tilt_stabilized",
               "tilt_age_s", "tilt_low", "tilt_high",
               "dial_mil", "dial_age_s", "fired_arc", "fired_by", "bearing_deg", "barrel_az_deg",
               "map", "grid_dz_m",
               "notes", "gun_source", "f2_age_s"]   # gun_source: F2 or minimap (2026-10-12)
POLL_MS = 15
F6_FAIL_S = 0.5          # holding F3 this long without one good read saves a debug capture
SAME_IMPACT = 0.05       # map units (5 m): an F4 this close to the last one...
SAME_IMPACT_S = 8.0      # ...and this soon after it is that shot again (a new shell takes longer)
TILT_EVERY_S = 0.25      # how often the tilt dial is read (a read holds the loop ~12 ms)
TILT_WEAPONS = {"spg"}   # the dial is the SPH-2's; the mortar sits on the ground
DIAL_MATCH = 15.0        # mil: a dial reading this close to an arc's shown mil is that arc, fired
DIAL_MAX_AGE_S = 120     # a dial reading older than this is not taken as what was fired
FIRE_FRESH_S = 3.0       # Space counts as a shot only with the dial or the tape read this recently (the gun
                         # view up; 1 s missed a live shot 2026-09-30 11:53; the dial alone missed one
                         # 2026-10-13 16:55 when the MIL scale had just dropped out)
HULL_TOL_DEG = 3.0       # a tape reading within this of hull heading + turret is that hull's
HULL_MOVED_DEG = 2.0     # pitch or roll this far from where the heading was anchored: the hull moved
HULL_LOST_READS = 8      # ...and this many readings in a row that are not: the hull turned, re-anchor
RIBBON = (860, 28, 840, 36)  # x, y, w, h on the frame: the top compass ribbon (recorded, not read yet)
# The tape's 15-degree step is taken on its own evidence only with CONFIRM_VIEWS views leading by
# CONFIRM_LEAD (the box is wrong 12% of reads: three views are enough to show a step, not to
# trust); a fresh minimap heading that lands on the step confirms it at once.
CONFIRM_VIEWS, CONFIRM_LEAD = 6, 1.5
GUN_MOVED_M = 15.0       # a minimap position this far from where the gun was set is a moved gun
TURRET_TOL_DEG = 12.0        # a tilt-dial turret read this far from turret_expected() is not taken...
TURRET_RESET_READS = 6       # ...unless this many in a row disagree (then the expectation was wrong)
TURRET_HIST_S = 1.5          # with nothing to expect, a read must agree with most reads of this long
MM_HEADING_FRESH_S = 0.75  # the minimap icon's hull heading is the bearing's trusted prior this long: one missed
                          # 0.25 s read, no more (review 2026-10-12: 2 s let a turning hull mislead it)
ICON_EVERY_S = 0.2        # the icon's heading is read at least this often while a minimap search runs
# Map choices that are another map's terrain and minimap read in SHIFTED map coordinates: (the
# terrain/minimap map, metres (east, north) to ADD to a minimap position to get this map's
# readout). TRAINING RANGE, 2026-10-14: all 10 shots from its pad (10-13 and 10-14, bearings
# 84-290, level hull, minimap-placed gun) landed as if the gun were 24 m west and 76 m south of
# the minimap's position (rms 8.5 m, the scatter); real Bakurani matches show no such shift.
MAP_VARIANTS = {"bakurani-training": ("bakurani", (-24.0, -76.0))}
MAP_CHOICES = (*terrain.MAPS, *MAP_VARIANTS)


def map_frame(m):
    """(the terrain/minimap map, (east, north) metres from a minimap position to m's readout)."""
    return MAP_VARIANTS.get(m, (m, (0.0, 0.0)))


AZ_BIAS_DEG = 1.0        # deg every shot is aimed right of the target before any F4: shells land left of where
                         # the barrel points -- 41 shots on 3 maps over 6 days, median 1.06 deg; 2026-10-13
                         # level hulls east 0.9/1.3 and south 0.8/0.6, so the gun's own, not one map direction
BEARING_MATCH = 5.0      # deg: a fired bearing this close to the shown azimuth corrects the next shot


class Reader:
    """Grabs the patch of screen the two labels sit in and reads it (main thread only)."""

    def __init__(self):
        import mss
        self.sct = mss.mss()
        self.virtual = self.sct.monitors[0]
        self.game = primary_monitor(self.sct.monitors)

    def read_tilt(self, need_turret=True):
        """The tilt dial at the bottom of the game screen: (hud_tilt.Tilt or None, why)."""
        x, y, w, h = hud_tilt.BOX
        crop = cv2.cvtColor(np.asarray(self.sct.grab(
            {"left": self.game["left"] + x, "top": self.game["top"] + y, "width": w, "height": h})),
            cv2.COLOR_BGRA2BGR)
        self.last_tilt = crop                    # recorded with the bearing (--record-bearing)
        return hud_tilt.read(crop, need_turret)

    def read_bearing(self):
        """The gun view's bearing tape as one observation for hud_bearing.StepTracker:
        (hud_bearing.TapeObs with the compass box's scores filled in, "") or (None, why)."""
        x, y, w, h = hud_bearing.BOX
        crop = cv2.cvtColor(np.asarray(self.sct.grab(
            {"left": self.game["left"] + x, "top": self.game["top"] + y, "width": w, "height": h})),
            cv2.COLOR_BGRA2BGR)
        obs, why = hud_bearing.observe(crop)
        self.last_tape, self.last_box = crop, None
        if obs is None:
            return None, why
        if hud_compass.has_samples():
            self.last_box = hud_compass.grab_box(self.sct, self.game)
            obs.compass, _ = hud_compass.totals(self.last_box, obs.candidates())
        return obs, ""

    def grab_ribbon(self):
        """The top compass ribbon (its white labels every 15 degrees and the box), for recording:
        USER 2026-10-12, they read where the tape's black labels cannot."""
        x, y, w, h = RIBBON
        return cv2.cvtColor(np.asarray(self.sct.grab(
            {"left": self.game["left"] + x, "top": self.game["top"] + y, "width": w, "height": h})),
            cv2.COLOR_BGRA2BGR)

    def read_mil(self):
        """The gun view's MIL scale: (hud_mil.Mil or None, why)."""
        x, y, w, h = hud_mil.BOX
        crop = cv2.cvtColor(np.asarray(self.sct.grab(
            {"left": self.game["left"] + x, "top": self.game["top"] + y, "width": w, "height": h})),
            cv2.COLOR_BGRA2BGR)
        self.last_mil = crop
        return hud_mil.read(crop)

    def read(self):
        cx, cy = cursor()
        dx, dy, w, h = map_coords.GRAB
        v = self.virtual
        left = max(v["left"], min(cx + dx, v["left"] + v["width"] - w))
        top = max(v["top"], min(cy + dy, v["top"] + v["height"] - h))
        crop = cv2.cvtColor(np.asarray(self.sct.grab(
            {"left": left, "top": top, "width": w, "height": h})), cv2.COLOR_BGRA2BGR)
        return (cx, cy), map_coords.read(crop, (cx, cy), origin=(left, top))

    def grab_minimap(self):
        """The minimap square of the game's screen (minimap.MM), BGR."""
        x, y, w, h = minimap.MM
        return cv2.cvtColor(np.asarray(self.sct.grab(
            {"left": self.game["left"] + x, "top": self.game["top"] + y, "width": w, "height": h})),
            cv2.COLOR_BGRA2BGR)

    def grab_game(self):
        """The game's whole screen, BGR."""
        return cv2.cvtColor(np.asarray(self.sct.grab(self.game)), cv2.COLOR_BGRA2BGR)

    def save_debug(self, at, why):
        """Whole virtual screen + the reason, named for the cursor point; written on a thread."""
        frame = cv2.cvtColor(np.asarray(self.sct.grab(self.virtual)), cv2.COLOR_BGRA2BGR)
        fx, fy = at[0] - self.virtual["left"], at[1] - self.virtual["top"]
        now = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
        base = os.path.join(DEBUG_DIR, f"map_{stamp}_at{fx}-{fy}")

        def write():
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(base + ".png", frame)
            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(f"cursor at frame point ({fx}, {fy})\n{why}\n")
        threading.Thread(target=write).start()
        return base + ".png"


def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _plain(o):
    if isinstance(o, (np.generic, np.ndarray)):         # a numpy number or array (the minimap's)
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def save_state(state):
    """Written whole to a temporary file and swapped in: a save that fails leaves the last one (live
    2026-10-12: a float32 failed json.dump halfway and the file was cut off at '"gun": [')."""
    tmp = STATE + ".tmp"
    try:
        text = json.dumps(state, default=_plain)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, STATE)
    except (OSError, TypeError, ValueError) as e:
        print(f"could not save {STATE}: {e}", flush=True)






def log_shot(row, path=None):
    """Append one row to shots.csv (header first when the file is new). Returns the path."""
    path = path or SHOTS
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != SHOT_FIELDS:          # written by an older version: add the new columns
            with open(path, newline="", encoding="utf-8") as f:
                old = list(csv.DictReader(f))
            tmp = path + ".tmp"                # written whole, then swapped in: a crash keeps the old file
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=SHOT_FIELDS)
                w.writeheader()
                for r in old:
                    w.writerow({k: r.get(k, "") for k in SHOT_FIELDS})
            os.replace(tmp, path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHOT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SHOT_FIELDS})
    return path


class App:
    """The window and its hotkeys. Drives itself with root.after(); a host program
    can instead call root.update() often and check `closed`."""

    def __init__(self, root, args, standalone=True):
        self.root, self.args, self.standalone = root, args, standalone
        self.closed = False
        self.weapons = ballistics.load_weapons()
        self.state = load_state()
        self.state["weapon"] = "spg"   # USER 2026-10-11: always the SPH-2, no weapon key
        if getattr(args, "map", None):
            self.state["map"] = args.map
        if self.state.get("map") not in MAP_CHOICES:
            self.state["map"] = None        # not picked yet: nothing is assumed about the map
        self.reader = Reader()
        self.k_gun, self.k_aim = Key(args.gun_key), Key(args.aim_key)
        self.k_impact = Key(args.impact_key)
        self.k_fire = Key(getattr(args, "fire_key", None) or vk_code("SPACE"))
        self.k_capture = Key(getattr(args, "capture_key", None) or vk_code("F5"))
        self.shot = None            # (hud_mil.Mil, time.time(), bearing or None) when Space fired
        self.fired_since_f8 = False # a Space shot since the last F4 that counted
        self.aim_since = None       # when the current F3 hold began
        self.aim_read_ok = False    # any good read during it
        self.aim_reported = False
        self.target = None
        self.solution = None
        self.adj = None             # {"target": (x, y), "az": deg, arc: mils} from F8s
        self.last_impact = None     # (x, y, time.time()) of the last F4 that counted
        self.use_tilt = not getattr(args, "no_tilt", False)
        self.tilt = None            # the last good read of the tilt dial (hud_tilt.Tilt)
        self.tilt_time = None       # time.time() of it
        self.tilt_why = "not read yet"
        self.dial = None            # the last good read of the MIL scale (hud_mil.Mil)
        self.dial_time = None
        self.dial_why = "not read yet"
        self.bearing = None         # the last good read of the bearing tape (degrees)
        self.bearing_time = None
        self.bearing_why = "not read yet"
        self.tracker = hud_bearing.StepTracker()   # decides the tape's 15-degree step over many reads
        self.locator = minimap.Locator()   # where the tank is, its hull heading and the map, off the minimap
        self.minimap_fix = None            # the last minimap.Fix taken, and when
        self.minimap_time = None
        self.minimap_why = "not read yet"
        self.icon_heading = None           # (the hull's heading off the minimap's icon, when): read on
                                           # every visible frame, placed or not (Zestafona rarely is)
        self.icon_read_t = float("-inf")   # time.monotonic() of the last icon read
        self.gun_source = "F2"             # what set the gun: "F2" or "minimap"
        self.map_source = "saved"          # what set the map: "saved", "button" (locks it) or "minimap"
        self.minimap_lock = None           # the map the button chose: the minimap searches no other
        self.minimap_gen = 0               # bumped by F2 and the button: older results are dropped
        self.gun_origin = self.state.get("gun")   # where the F4 correction and the hull anchor belong
        # The minimap is matched on a worker thread: a whole-map search takes ~0.4 s, and a loop
        # held that long misses a tapped F3/F4 (keys are polled). None runs it inline (tests).
        self.minimap_pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="minimap")
        self.minimap_job = None
        self.minimap_seed = None           # (map, x, y): F2's position, handed to the worker
        session = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_shots",
                               "bearing_live", time.strftime("%Y%m%d-%H%M%S"))
        recording = getattr(args, "record_bearing", False)
        self.recorder = hud_bearing.Recorder(session) if recording else None
        self.mil_recorder = hud_bearing.Recorder(os.path.join(session, "mil")) if recording else None
        self.mm_recorder = hud_bearing.Recorder(os.path.join(session, "minimap")) if recording else None
        self.hull_heading = None    # the hull's compass heading = tape - turret, constant while it sits
        self.hull_proven = False    # ...taken while the step was proven: it proves the step until moved
        self.hull_off = 0           # readings in a row that disagreed with it
        self.hull_attitude = None   # (pitch, roll) when it was anchored
        self.next_tilt = 0.0
        self.bearing_ok = self.dial_ok = self.tilt_ok = False   # read on the last poll
        self.turret_misses = 0      # turret reads in a row that disagreed with turret_expected()
        self.turret_hist = []       # (time, turret, tape) of every recent turret read, taken or not
        self.tilt_tape = None       # the tape when the last good tilt was read

        big, mid, small = ("Segoe UI", 20, "bold"), ("Segoe UI", 13), ("Segoe UI", 10)
        root.title("Artillery")
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.configure(bg="#111")
        self.vars = {k: tk.StringVar() for k in ("weapon", "gun", "tilt", "target", "range", "dz",
                                                  "mils", "dial", "bearing", "status")}
        # the small grey facts on top, the numbers to act on below them
        for key, font, fg, pad in (("weapon", small, "#9ad", 0), ("gun", small, "#999", 0),
                                   ("target", small, "#999", 0), ("dz", small, "#fd6", 0),
                                   ("tilt", small, "#999", 0),
                                   ("range", big, "#fff", 2), ("mils", big, "#6f6", 2),
                                   ("dial", big, "#6cf", 2), ("bearing", big, "#6cf", 2),
                                   ("status", mid, "#fa6", 2)):
            tk.Label(root, textvariable=self.vars[key], font=font, fg=fg, bg="#111",
                     anchor="w", justify="left").pack(fill="x", padx=14, pady=pad)
        self.map_label = tk.StringVar()
        tk.Button(root, textvariable=self.map_label, command=self.cycle_map, font=("Segoe UI", 10),
                  bg="#222", fg="#ddd", relief="flat").pack(anchor="w", padx=14, pady=(2, 8))
        self.render()
        root.after(POLL_MS, self.tick)

    def close(self):
        """The window's X. Standalone that ends the program; a host program sees `closed`."""
        self.save_position()
        self.shutdown()
        self.closed = True
        if self.standalone:
            self.root.destroy()

    # --- display ------------------------------------------------------------------
    def save_position(self):
        """Remember where the window is, for the next time it opens (artillery_state.json)."""
        self.state["window"] = [self.root.winfo_x(), self.root.winfo_y()]
        save_state(self.state)

    def render(self, status=None):
        w = self.weapons[self.state["weapon"]]
        g = self.state.get("gun")
        self.vars["weapon"].set(f"{w.name}   F2 gun   F3 (hold) target   F4 impact   F5 minimap")
        if not g:
            self.vars["gun"].set("Gun     not set -- F2 with the cursor on it")
        else:
            age = ""
            if self.gun_source == "minimap" and self.minimap_time is not None and \
                    time.monotonic() - self.minimap_time >= 5:
                age = f", {time.monotonic() - self.minimap_time:.0f} s ago: {self.minimap_why or 'not placed'}"
            self.vars["gun"].set(f"Gun     x{g[0]:.2f}  y{g[1]:.2f}   ({self.gun_source}{age})")
        self.vars["tilt"].set(self.tilt_text())
        t = self.target
        self.vars["target"].set(f"Target  x{t[0]:.2f}  y{t[1]:.2f}" if t else "Target  hold F3 over it")
        self.vars["dz"].set(self.grid_text() if t else "")
        src = {"minimap": "(minimap)", "button": "(locked - click to change)"}.get(self.map_source,
                                                                                "(click to change)")
        if "no map cache" in self.minimap_why:
            src += "   minimap: python minimap.py --build"
        self.map_label.set(f"Map: {self.state['map'] or 'not picked'}   {src}")
        s = self.solution
        if s is None:
            self.vars["range"].set("")
            self.vars["mils"].set("")
        else:
            adj = self.adjustment()
            az = self.aim_az(s)
            self.vars["range"].set(f"{ballistics.js_round(s.distance_m)} m    az {az % 360:.1f}°"
                                   + (f" (adjusted {adj['az']:+.1f}°)" if adj and adj["az"] else ""))
            if not s.in_range:
                self.vars["mils"].set(f"out of range ({w.min_m:.0f}–{w.max_m:.0f} m)")
            else:
                lines = []
                for arc, m in s.mils.items():
                    name = "" if arc == "single" else f"{arc.upper():<5}"
                    shown = self.shown_mil(s, arc)
                    lines.append(f"{name}{ballistics.js_round(shown) if shown is not None else (ballistics.format_mil(m) or '--')}")
                self.vars["mils"].set("\n".join(lines))
        self.vars["dial"].set(self.dial_text())
        self.vars["bearing"].set(self.bearing_text())
        if status is not None:
            self.vars["status"].set(status)

    def cycle_map(self):
        """The window's map button: none -> bakurani -> ozeti -> zestafona -> bakurani-training -> none."""
        order = [None, *MAP_CHOICES]
        self.state["map"] = order[(order.index(self.state["map"]) + 1) % len(order)]
        # the button locks the map: the minimap searches only it ("none" frees it again), and a
        # result from before the press is dropped. Not a moved gun -- the F4 correction stays.
        self.map_source = "button" if self.state["map"] else "saved"
        self.minimap_lock = self.state["map"]
        self.minimap_gen += 1
        save_state(self.state)
        self.render("")

    def grid_text(self):
        m = self.state.get("map")
        if m is None:
            return "Height  pick the map below -- the mils assume level ground until then"
        if not (self.state.get("gun") and self.target):
            return ""
        d = self.dz()
        if d is None:
            return f"Height  no {m} terrain data for these two spots -- level ground assumed"
        where = "target higher" if d > 0 else "target lower" if d < 0 else "level"
        return f"Height  {d:+.0f} m ({where})  [{m} terrain, 2 m grid]"

    def capture_minimap(self):
        """F5: save the game screen (the map closed, so the minimap shows) with the gun position
        F2 last stored, to MINIMAP_DIR, a PNG and a row of log.csv each; written on a thread."""
        g = self.state.get("f2_gun") or self.state.get("gun")
        if not g:
            self.render("F5 needs the gun stored: map open, F2 on your tank, close the map, F5")
            beep(400, 150)
            return
        frame = self.reader.grab_game()
        now = time.time()
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}.png"
        t, b = self.tilt, self.bearing
        r1 = lambda v: "" if v is None or v != v else round(v, 2)
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), "map": self.state.get("map") or "",
               "gun_x": g[0], "gun_y": g[1],
               "gun_age_s": round(now - self.state.get("f2_time", self.state.get("gun_time", now)), 1),
               "tilt_pitch_deg": r1(t.pitch_deg if t else None), "tilt_roll_deg": r1(t.roll_deg if t else None),
               "tilt_turret_deg": r1(t.turret_deg if t else None), "bearing_deg": r1(b),
               "hull_heading_deg": r1(self.hull_heading), "file": name, "gun_source": self.gun_source,
               "mm_x": r1(self.minimap_fix.x if self.minimap_fix else None),
               "mm_y": r1(self.minimap_fix.y if self.minimap_fix else None),
               "mm_heading": r1(self.minimap_fix.heading if self.minimap_fix else None)}

        def write():
            os.makedirs(MINIMAP_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(MINIMAP_DIR, name), frame)
            log = os.path.join(MINIMAP_DIR, "log.csv")
            new = not os.path.exists(log)
            if not new:                          # an older header: rewrite it with the new columns
                with open(log, encoding="utf-8") as f:
                    old = list(csv.DictReader(f))
                    f.seek(0)
                    header = f.readline().strip().split(",")
                if header != MINIMAP_FIELDS:
                    tmp = log + ".tmp"
                    with open(tmp, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=MINIMAP_FIELDS)
                        w.writeheader()
                        w.writerows({k: r.get(k, "") for k in MINIMAP_FIELDS} for r in old)
                    os.replace(tmp, log)
            with open(log, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=MINIMAP_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
        threading.Thread(target=write, name="minimap-save").start()
        self.render(f"minimap saved with the gun at x{g[0]:.2f} y{g[1]:.2f} "
                    f"({row['gun_age_s']:.0f} s after F2)")
        print(f"minimap: {name} gun x{g[0]:.2f} y{g[1]:.2f}", flush=True)
        beep(1500, 60)

    def mark_impact(self):
        """F4: read where the shell landed, show the miss, log the shot."""
        g, t, s = self.state.get("gun"), self.target, self.solution
        if not (g and t and s):
            self.render("F4 needs a target first: hold F3 over it")
            beep(400, 150)
            return
        at, r = self.reader.read()
        if not r.ok:
            path = self.reader.save_debug(at, r.why)
            self.render(f"could not read the map there: {r.why}")
            print(f"F4: {r.why} -- saved {path}", flush=True)
            beep(400, 150)
            return
        long_m, right_m = ballistics.miss(g, t, (r.x, r.y))
        li = self.last_impact
        if (self.adjustment() and li and math.hypot(r.x - li[0], r.y - li[1]) <= SAME_IMPACT
                and (time.time() - li[2] <= SAME_IMPACT_S or not self.fired_since_f8)):
            # Live 2026-09-29 21:51: ten F8s on one crater stacked ten corrections; 2026-09-30
            # 11:52 one crater marked again 23 s later. A new shell there comes after a Space.
            self.render(f"same impact as the last F4 ({abs(long_m):.0f} m "
                        f"{'long' if long_m >= 0 else 'short'}) -- already counted, not logged again")
            beep(700, 60)
            return
        weapon = self.weapons[self.state["weapon"]]
        hit = ballistics.solve(weapon, g, (r.x, r.y))
        old = self.adjustment() or {}
        shown = {arc: self.shown_mil(s, arc) for arc in s.mils}
        tilt_d = {arc: self.tilt_delta(self.world_mil(s, arc)) for arc in s.mils}
        fired = self.fired(s)                   # (arc, dial - shown) or None
        shown_az = self.aim_az(s)
        # Each shot says what the adjustment should have been for it: what it was fired on
        # (the dial and bearing when known, else what was shown) plus its miss. The next
        # shot uses the MEAN of those for this target -- one shell scatters ~8 m (10 shots
        # 2026-09-30), so the last shot alone chases the scatter.
        est = {k: list(v) for k, v in old.get("est", {}).items()}
        new = {"target": t, "az": old.get("az", 0) - math.degrees(math.atan2(right_m, s.distance_m))}
        fb = self.shot_bearing()
        if fb is not None:
            fb = self.barrel_azimuth(fb)
        if fb is not None and abs((fb - shown_az + 180) % 360 - 180) <= BEARING_MATCH:
            new["az"] += (fb - shown_az + 180) % 360 - 180     # from where it pointed, not what was shown
        est.setdefault("az", []).append(new["az"])
        for arc in s.mils:
            if fired and fired[0] != arc:
                new[arc] = old.get(arc, 0)      # this shot says nothing about an arc not fired
                continue
            d = ballistics.adjust(weapon, arc, s.distance_m, self.dz(), long_m)
            e = old.get(arc, 0) + (d or 0)
            if fired:
                e += fired[1]                   # correct from what was fired, not what was shown
            est.setdefault(arc, []).append(e)
        for k, v in est.items():
            new[k] = sum(v) / len(v)
        new["est"] = est
        self.adj = new
        self.fired_since_f8 = False
        self.last_impact = (r.x, r.y, time.time())
        nxt = {arc: self.shown_mil(s, arc) for arc in s.mils}
        r1 = lambda v: "" if v is None else round(v, 1)
        mils = {arc: (round(m.mil, 1) if m and m.mil is not None else "") for arc, m in s.mils.items()}
        gt = self.state.get("gun_time")
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "weapon": self.state["weapon"],
            "gun_x": g[0], "gun_y": g[1],
            "gun_age_s": "" if gt is None else round(time.time() - gt),
            "gun_source": self.gun_source,
            "f2_age_s": "" if self.state.get("f2_time") is None else round(time.time() - self.state["f2_time"]),
            "target_x": t[0], "target_y": t[1],
            "dz_m": "" if self.dz() is None else round(self.dz(), 1),
            "distance_m": round(s.distance_m, 1), "azimuth_deg": round(s.azimuth_deg, 2),
            "mil_single": mils.get("single", ""), "mil_low": mils.get("low", ""),
            "mil_high": mils.get("high", ""),
            "mil_low_height": (round(s.height["low"][0], 1)
                               if "low" in s.height and s.height["low"][0] is not None else ""),
            "shown_low": r1(shown.get("low")), "shown_high": r1(shown.get("high")),
            "shown_single": r1(shown.get("single")), "shown_az": round(shown_az % 360, 2),
            "next_low": r1(nxt.get("low")), "next_high": r1(nxt.get("high")),
            "next_single": r1(nxt.get("single")), "next_az": round((s.azimuth_deg + AZ_BIAS_DEG + new["az"]) % 360, 2),
            "impact_x": r.x, "impact_y": r.y, "long_m": round(long_m, 1), "right_m": round(right_m, 1),
            "impact_distance_m": round(hit.distance_m, 1), "impact_azimuth_deg": round(hit.azimuth_deg, 2),
        }
        tl = self.tilt if self.tilt_applies() else None
        if tl is not None:
            row.update({
                "tilt_pitch_ticks": round(tl.pitch_ticks, 2), "tilt_pitch_deg": round(tl.pitch_deg, 1),
                "tilt_roll_deg": round(tl.roll_deg, 1), "tilt_turret_deg": round(tl.turret_deg, 1),
                "tilt_stabilized": "yes" if tl.stabilized else "no",
                "tilt_age_s": round(time.time() - self.tilt_time),
                "tilt_low": r1(tilt_d.get("low")), "tilt_high": r1(tilt_d.get("high")),
            })
        sd = self.shot_dial()
        if sd is not None:
            row.update({"dial_mil": round(sd[0].mil, 2), "dial_age_s": round(time.time() - sd[1]),
                        "fired_arc": fired[0] if fired else "", "fired_by": sd[2]})
        if fb is not None:
            row["barrel_az_deg"] = round(fb, 2)
            if self.shot_bearing() is not None:
                row["bearing_deg"] = round(self.shot_bearing(), 2)
        row["map"] = self.state.get("map") or ""
        row["grid_dz_m"] = row["dz_m"]
        self.shot = None                        # the next F4 is for the next shot
        path = log_shot(row)
        words = (f"{abs(long_m):.0f} m {'long' if long_m >= 0 else 'short'}, "
                 f"{abs(right_m):.0f} m {'right' if right_m >= 0 else 'left'}")
        self.render(f"impact: {words} -- logged; the next shot's mils and azimuth are adjusted above")
        print(f"impact x{r.x:.2f} y{r.y:.2f}: {words} -> {path}", flush=True)
        beep(1200, 60)


    def aim_az(self, s):
        """The azimuth to lay for solution `s`: the target's, AZ_BIAS_DEG right, plus the F4
        correction (which is learned on top of the bias). Not wrapped to 0-360."""
        adj = self.adjustment()
        return s.azimuth_deg + AZ_BIAS_DEG + (adj["az"] if adj else 0)

    def adjustment(self):
        """The F4 adjustment for this gun position (every target until F2), or None."""
        return self.adj

    def world_mil(self, s, arc):
        """The elevation the shot needs, in mil above level: height-corrected where it
        can be, plus any F4 adjustment for this target. None when there is none."""
        mil = None
        if arc in s.height and s.height[arc][0] is not None:
            mil = s.height[arc][0]
        elif s.mils.get(arc) is not None and s.mils[arc].mil is not None:
            mil = s.mils[arc].mil
        adj = self.adjustment()
        return None if mil is None else mil + (adj.get(arc, 0) if adj else 0)

    def shown_mil(self, s, arc):
        """The mil the window shows for an arc: world_mil as set on the dial of a gun
        whose hull sits the way the tilt dial last showed. None when there is none."""
        mil = self.world_mil(s, arc)
        d = self.tilt_delta(mil)
        return None if mil is None else mil + (d or 0)

    def tilt_applies(self):
        return self.use_tilt and self.state["weapon"] in TILT_WEAPONS and self.tilt is not None

    def tilt_delta(self, mil):
        """Mils the tilt adds on the dial for an elevation of `mil`; None when there is no
        correction (no reading, --no-tilt, the mortar) or the tilted gun cannot reach it."""
        if mil is None or not self.tilt_applies():
            return None
        t = self.tilt
        return ballistics.tilt_mil(mil, t.pitch_deg, t.roll_deg, t.turret_deg)

    def tilt_text(self):
        if self.state["weapon"] not in TILT_WEAPONS:
            return ""
        if not self.use_tilt:
            return "Tilt    off (--no-tilt): the mils assume a level hull"
        t = self.tilt
        if t is None:
            return f"Tilt    {self.tilt_why} -- the dial shows in the gun seat with the map closed"
        age = time.time() - self.tilt_time
        return (f"Tilt    nose {'up' if t.pitch_deg >= 0 else 'down'} {abs(t.pitch_deg):.1f}° "
                f"({t.pitch_ticks:+.2f} tick), {'right' if t.roll_deg >= 0 else 'left'} side up "
                f"{abs(t.roll_deg):.1f}°, turret {abs(t.turret_deg):.0f}° "
                f"{'right' if t.turret_deg >= 0 else 'left'}, "
                f"{'Stabilized' if t.stabilized else 'Unstabilized'}   ({age:.0f} s ago)")

    def shot_bearing(self):
        """Where the gun pointed for the last shot: the bearing as Space was pressed, or None.
        Never a later reading: live 2026-10-13 16:55 a shot fired unseen was corrected from the
        tape 4.2 deg off where it had fired (the turret turned after), and every shot after
        went ~2 deg left until the next F2."""
        return self.shot[2] if self.shot is not None else None

    def barrel_azimuth(self, tape):
        """The world azimuth the barrel pointed when the tape read `tape`: the tape is the
        turret's heading, and a rolled hull swings an elevated barrel off it. The tape itself
        with no tilt reading or no dial."""
        sd = self.shot_dial()
        t = self.tilt
        if not self.tilt_applies() or sd is None or t.turret_deg != t.turret_deg:
            return tape
        return ballistics.barrel(tape - t.turret_deg, t.pitch_deg, t.roll_deg, t.turret_deg, sd[0].mil)[0]

    def poll_minimap(self):
        """Read the minimap (minimap.Locator) -- on the worker thread: the crop is grabbed here (mss
        belongs to this thread), the match runs there, and its result is taken on a later poll.
        The icon's hull heading is read here too, every ICON_EVERY_S even while a search runs:
        live 2026-10-13 a whole-map search took 0.4-0.7 s, and a heading that waited for it was
        stale (MM_HEADING_FRESH_S). A grab and read is ~12 ms, so not on every tick."""
        busy = self.minimap_job is not None and not self.minimap_job.done()
        now = time.monotonic()
        if busy and now - self.icon_read_t < ICON_EVERY_S:
            return
        crop = self.reader.grab_minimap()
        self.icon_read_t = now
        heading = minimap.icon_heading(crop)[0]
        if heading is not None:
            self.icon_heading = (heading, now)
        if busy:
            return
        if self.minimap_job is not None:
            job, self.minimap_job = self.minimap_job, None
            self.take_result(job)
        seed, self.minimap_seed = self.minimap_seed, None
        self.minimap_crop = crop                 # recorded with its result (--record-bearing)
        gen, lock, loc = self.minimap_gen, self.minimap_lock, self.locator

        def work():                              # holds the locator only, never the window (Tk
            now = time.monotonic()               # objects must not be freed on the worker thread)
            loc.only_map = map_frame(lock)[0] if lock else None
            if seed:
                loc.seed(*seed, now)
            return loc.locate(crop, now) + (now, gen, heading)
        if self.minimap_pool is None:
            f = concurrent.futures.Future()
            try:
                f.set_result(work())
            except Exception as e:               # noqa: BLE001 -- shown, never fatal to the window
                f.set_exception(e)
            self.take_result(f)
        else:
            self.minimap_job = self.minimap_pool.submit(work)

    def take_result(self, job):
        """A finished read: an error is a reason (the window goes on); a read from before F2 or
        the map button, or one held up by a busy spell, is dropped."""
        try:
            fix, why, now, gen, heading = job.result()
        except Exception as e:                   # noqa: BLE001
            self.minimap_why = f"minimap error: {e}"
            return
        taken = gen == self.minimap_gen and time.monotonic() - now <= 1.5
        if self.mm_recorder:
            self.mm_recorder.save({"minimap": getattr(self, "minimap_crop", None)},
                                  {"t": now, "wall": time.time(), "taken": taken, "why": why,
                                   "map": fix.map if fix else None, "x": fix.x if fix else None,
                                   "y": fix.y if fix else None, "score": fix.score if fix else None,
                                   "lead": fix.lead if fix else None, "icon_heading": heading})
        if not taken:
            return
        if fix is not None and self.minimap_lock:
            src, (sx, sy) = map_frame(self.minimap_lock)
            if fix.map != src:
                return
            # the locked map's own coordinates (a variant's readout is shifted from the minimap's)
            fix = dataclasses.replace(fix, map=self.minimap_lock, x=fix.x + sx / 100, y=fix.y + sy / 100)
        self.take_minimap(fix, why, now)

    def step_proven(self):
        """The tape's 15-degree step backed by more than a hull anchor that may be stale: a fresh
        minimap heading plus the turret angle lands on it, the tape's black labels proved it (read
        clearly over a bright sky, USER 2026-10-12), or the tape's own evidence confirms it."""
        if self.tracker.label_proven:
            return True
        ih, t = self.icon_heading, self.tilt
        if (ih is not None and self.bearing is not None and t is not None and t.turret_deg == t.turret_deg
                and time.monotonic() - ih[1] <= MM_HEADING_FRESH_S
                and abs((self.bearing - ih[0] - t.turret_deg + 180) % 360 - 180) < 7.5):
            return True
        return self.tracker.confirmed(CONFIRM_VIEWS, CONFIRM_LEAD)

    def shutdown(self):
        """Stop the minimap's worker thread (the window is closing)."""
        if self.minimap_pool is not None:
            self.minimap_pool.shutdown(wait=False, cancel_futures=True)

    def take_minimap(self, fix, why, now):
        """A confident fix is the gun's position and the map; a move of more than GUN_MOVED_M from
        where the gun was set is a moved gun, which drops the F4 correction and the hull anchor as
        F2 does -- jitter of a metre or two does not."""
        if fix is None:
            self.minimap_why = why
            return
        self.minimap_fix, self.minimap_time, self.minimap_why = fix, now, ""
        g = [round(float(fix.x), 2), round(float(fix.y), 2)]   # plain floats: the minimap's are float32
        o = self.gun_origin
        new_map = fix.map != self.state.get("map")
        # a move from where the gun was set; the very first fix (no origin yet) moves nothing
        moved = o is not None and (new_map or math.hypot(g[0] - o[0], g[1] - o[1]) * 100 > GUN_MOVED_M)
        if new_map:
            self.state["map"], self.map_source = fix.map, "minimap"
        if o is None:
            self.gun_origin = g
        self.state["gun"], self.state["gun_time"], self.gun_source = g, time.time(), "minimap"
        if moved:
            self.gun_origin = g
            self.adj = None
            self.hull_heading, self.hull_proven = None, False
            save_state(self.state)
        if self.target:
            self.solve()

    def poll_bearing(self):
        """Read the bearing tape and let hud_bearing.StepTracker decide its 15-degree step. The
        prior for a fresh start is the hull's heading plus the turret angle on the tilt dial when
        anchored (trusted: the hull does not turn while the gun does), else the target's azimuth
        (untrusted: the window shows "settling" until the evidence agrees). The anchor is only
        ever taken from a settled reading."""
        turret = self.tilt.turret_deg if (self.use_tilt and self.tilt is not None) else float("nan")
        fresh = self.tilt is not None and time.time() - self.tilt_time <= 2.0 and turret == turret
        if fresh and self.hull_attitude is not None and (
                abs(self.tilt.pitch_deg - self.hull_attitude[0]) > HULL_MOVED_DEG
                or abs(self.tilt.roll_deg - self.hull_attitude[1]) > HULL_MOVED_DEG):
            self.hull_heading, self.hull_proven = None, False   # it sits differently now: driven
        ih = self.icon_heading
        if ih is not None and fresh and time.monotonic() - ih[1] <= MM_HEADING_FRESH_S:
            prior, trusted = (ih[0] + turret) % 360, True           # the hull off the minimap's icon
        elif self.hull_heading is not None and fresh:
            prior, trusted = (self.hull_heading + turret) % 360, True
        elif self.solution is not None:
            prior, trusted = self.aim_az(self.solution) % 360, False
        else:
            prior, trusted = None, False
        self.bearing_ok = False                 # read and settled THIS time
        obs, why = self.reader.read_bearing()
        if obs is None:
            self.bearing_why = why
            if self.recorder:                    # the moments worth looking at again
                self.recorder.save({"tape": getattr(self.reader, "last_tape", None)},
                                   {"t": time.monotonic(), "wall": time.time(), "prior": prior,
                                    "trusted": trusted, "deg": None, "state": "notape", "why": why})
            return
        tr = self.tracker.update(time.monotonic(), obs.frac, obs.evidence(), prior, trusted, obs.proof)
        if self.recorder:
            mf = self.minimap_fix
            # the tilt dial too: live 2026-10-14 its turret read +124 for minutes while the minimap's
            # hull heading disagreed with tape - turret by ~100 deg, and nothing showed why
            self.recorder.save({"tape": self.reader.last_tape, "box": self.reader.last_box,
                                "ribbon": self.reader.grab_ribbon(),      # for a ribbon reader to come
                                "tilt": getattr(self.reader, "last_tilt", None)},
                               {"t": time.monotonic(), "wall": time.time(), "prior": prior, "trusted": trusted,
                                "deg": tr.deg, "state": tr.state, "turret": turret if fresh else None,
                                "minimap": None if mf is None else {"map": mf.map, "x": mf.x, "y": mf.y,
                                                                    "heading": mf.heading,
                                                                    "age_s": time.monotonic() - self.minimap_time},
                                "icon_heading": None if ih is None else ih[0]})
        if tr.deg is None:
            self.bearing, self.bearing_why = None, tr.why
            return
        self.bearing, self.bearing_time = tr.deg, time.time()
        self.bearing_ok = True
        if not fresh:
            return
        seen = (tr.deg - turret) % 360
        if self.hull_heading is None or tr.state == "switched":
            # a corrected step re-anchors at once: left for HULL_LOST_READS, a map opened in that
            # time brought the wrong heading back as trusted (review 2026-10-11)
            self.hull_heading, self.hull_off = seen, 0
            self.hull_attitude = (self.tilt.pitch_deg, self.tilt.roll_deg)
            self.hull_proven = self.step_proven()
            return
        diff = (seen - self.hull_heading + 180) % 360 - 180
        if abs(diff) <= HULL_TOL_DEG:
            self.hull_heading = (self.hull_heading + 0.2 * diff) % 360
            self.hull_off = 0
            self.hull_proven = self.hull_proven or self.step_proven()
        else:
            self.hull_off += 1
            if self.hull_off >= HULL_LOST_READS:     # the hull really turned: take the reading
                self.hull_heading, self.hull_off = seen, 0
                self.hull_proven = self.step_proven()

    def bearing_text(self):
        """How far to traverse, signed (+ right): nothing else, the bearing itself is on the tape."""
        if self.state["weapon"] not in TILT_WEAPONS:
            return ""
        if self.bearing is None:
            if self.bearing_why.startswith("settling"):
                return "Bearing settling… turn the turret a little" if "turn" in self.bearing_why \
                    else "Bearing settling…"
            return f"Bearing {self.bearing_why}"
        s = self.solution
        if s is None:
            return ""
        age = time.time() - self.bearing_time
        stale = f"   ({age:.0f} s ago)" if age > 2 else ""
        want = self.bearing_goal(s)
        turn = (want - self.bearing + 180) % 360 - 180
        return f"Bearing {turn:+.1f}°{stale}"

    def bearing_goal(self, s):
        """The tape bearing to traverse to. On a level hull that is the shot's azimuth; on a
        rolled one the barrel swings off the turret's heading as it elevates (roll 12.6 deg,
        barrel 11 deg: 2.4 deg, live 2026-09-30), so the tape is turned the other way by
        exactly that. Falls back to the plain azimuth without a fresh tilt and bearing."""
        az = self.aim_az(s) % 360
        if not (self.tilt_applies() and self.bearing is not None and s.in_range and s.mils):
            return az
        t = self.tilt
        if t.turret_deg != t.turret_deg:                      # nan: the barrel was not readable
            return az
        arcs = [(a, self.world_mil(s, a)) for a in s.mils]
        arcs = [(a, m) for a, m in arcs if m is not None]
        if not arcs:
            return az
        d = self.dial.mil if self.dial is not None else None
        arc, mil = min(arcs, key=lambda am: abs(am[1] - d) if d is not None else am[1])   # the arc being set
        heading = self.bearing - t.turret_deg
        got = ballistics.aim(heading, t.pitch_deg, t.roll_deg, az, mil, turret0=t.turret_deg)
        return az if got is None else (heading + got[0]) % 360

    def shot_dial(self):
        """(hud_mil.Mil, its time, "space" or "last reading") of the dial the last shot was
        fired on, or None: the dial as Space was pressed, else the last reading."""
        if self.shot is not None:
            return (self.shot[0], self.shot[1], "space") if self.shot[0] is not None else None
        if self.dial is not None:
            return self.dial, self.dial_time, "last reading"
        return None

    def fired(self, s):
        """(arc, dial - shown mil) for the arc whose shown mil the shot's dial sits within
        DIAL_MATCH of, or None: no recent reading, or the dial is on no arc."""
        sd = self.shot_dial()
        if sd is None or time.time() - sd[1] > DIAL_MAX_AGE_S:
            return None
        d = sd[0].mil
        best = None
        for arc in s.mils:
            shown = self.shown_mil(s, arc)
            if shown is not None and abs(d - shown) <= DIAL_MATCH:
                if best is None or abs(d - shown) < abs(best[1]):
                    best = (arc, d - shown)
        return best

    def dial_text(self):
        """The MIL scale: for each arc, how far the dial still has to turn (+ up), one per line."""
        if self.state["weapon"] not in TILT_WEAPONS:
            return ""
        if self.dial is None:
            return f"Dial    {self.dial_why}"
        s = self.solution
        if s is None or not s.in_range:
            return ""
        d = self.dial.mil
        lines = []
        for arc in s.mils:
            m = self.shown_mil(s, arc)
            if m is not None:
                lines.append(f"{'' if arc == 'single' else f'{arc.upper():<5}'}{m - d:+.1f}")
        if lines:
            age = time.time() - self.dial_time
            if age > 2:
                lines[-1] += f"   ({age:.0f} s ago)"
        return "\n".join(lines)

    def poll_dial(self):
        m, why = self.reader.read_mil()
        self.dial_ok = m is not None            # read THIS time
        if m is not None:
            self.dial, self.dial_time = m, time.time()
        self.dial_why = why or self.dial_why
        if self.mil_recorder:
            self.mil_recorder.save({"strip": self.reader.last_mil},
                                   {"t": time.time(), "mil": m.mil if m else None,
                                    "labels": m.labels if m else 0, "why": why})

    def poll_all(self):
        self.poll_dial()
        if self.use_tilt:
            self.poll_tilt(render=False)           # the turret angle first: it anchors the tape
        self.poll_minimap()                        # the hull heading next: the tape's trusted prior
        self.poll_bearing()

    def poll_tilt(self, render=True):
        t, why = self.reader.read_tilt()
        if t is not None and t.turret_deg == t.turret_deg:
            off = lambda a, b: abs((a - b + 180) % 360 - 180)      # noqa: E731
            exp = self.turret_expected()
            now = time.time()
            tape = self.bearing if (self.bearing_ok and self.bearing is not None) else None
            recent = [h for h in self.turret_hist if now - h[0] <= TURRET_HIST_S]
            self.turret_hist = recent + [(now, t.turret_deg, tape)]   # every read, taken or not
            if exp is not None:
                bad = off(t.turret_deg, exp) > TURRET_TOL_DEG
                self.turret_misses = self.turret_misses + 1 if bad else 0
                bad = bad and self.turret_misses < TURRET_RESET_READS
            else:
                # no reference yet: the recent reads vote, each turned on by the tape since
                sup = sum(off(tu + (tape - tp if tape is not None and tp is not None else 0), t.turret_deg)
                          <= TURRET_TOL_DEG for _, tu, tp in recent)
                held = self.tilt is not None and self.tilt.turret_deg == self.tilt.turret_deg and \
                    off(self.tilt.turret_deg, t.turret_deg) <= TURRET_TOL_DEG
                bad = bool(recent) and not (sup > len(recent) - sup
                                            or (2 * sup == len(recent) and (held or self.tilt is None)))
            if bad:
                t, why = None, (f"the tilt dial's turret read {t.turret_deg:+.0f}, "
                                f"{'expected %+.0f' % exp if exp is not None else 'unlike the reads before'}")
        self.tilt_ok = t is not None and t.turret_deg == t.turret_deg     # read THIS time, turret too
        if t is not None:
            self.tilt, self.tilt_time = t, time.time()
            self.tilt_tape = self.bearing if self.bearing_ok else None
        self.tilt_why = why or self.tilt_why
        if render:
            self.render()



    def turret_expected(self):
        """Where the turret should read, or None when nothing says (live 2026-10-12 09:06: the dial
        read 145 on a quarter of reads, true ~3 -- something behind the see-through dial taken for
        the barrel). Turret = tape - hull heading: off a fresh minimap heading, else a proven hull
        anchor; else the last good turret turned as far as the tape has since."""
        if self.bearing is None or time.time() - self.bearing_time > 0.6:
            return None
        ih = self.icon_heading
        if ih is not None and time.monotonic() - ih[1] <= MM_HEADING_FRESH_S:
            return (self.bearing - ih[0] + 180) % 360 - 180
        if self.hull_proven and self.hull_heading is not None:
            return (self.bearing - self.hull_heading + 180) % 360 - 180
        t, tape = self.tilt, getattr(self, "tilt_tape", None)
        if (t is not None and tape is not None and t.turret_deg == t.turret_deg
                and time.time() - self.tilt_time <= 1.5):
            return (t.turret_deg + (self.bearing - tape) + 180) % 360 - 180
        return None

    def dz(self):
        """Target height minus gun height from the terrain grid, in metres; None while the
        map, the gun or the target is not set, or a spot is outside the grid. Both spots are
        first moved by the map's measured map-to-grid offset (terrain.MAP_OFFSET_M), and a
        variant's (MAP_VARIANTS) back onto its source map's coordinates."""
        m, g, t = self.state.get("map"), self.state.get("gun"), self.target
        if not (m and g and t):
            return None
        src, (sx, sy) = map_frame(m)
        ox, oy = terrain.MAP_OFFSET_M.get(src, (0.0, 0.0))
        ox, oy = ox - sx, oy - sy
        return terrain.grid_dz(src, (g[0] + ox / 100, g[1] + oy / 100), (t[0] + ox / 100, t[1] + oy / 100))

    def solve(self):
        g = self.state.get("gun")
        if g and self.target:
            self.solution = ballistics.solve(self.weapons[self.state["weapon"]], g, self.target, self.dz())

    # --- the loop -----------------------------------------------------------------
    def tick(self):
        if self.closed:
            return
        try:
            self.step()
        finally:
            self.root.after(POLL_MS, self.tick)

    def step(self):
        now = time.perf_counter()
        if self.state["weapon"] in TILT_WEAPONS and now >= self.next_tilt:
            self.next_tilt = now + TILT_EVERY_S     # the MIL scale is read on the same beat (~8 ms)
            self.poll_all()
            self.render()
        fresh_d = self.dial is not None and time.time() - self.dial_time <= FIRE_FRESH_S
        fresh_b = self.bearing is not None and time.time() - self.bearing_time <= FIRE_FRESH_S
        if self.k_fire.pressed():
            if fresh_d or fresh_b:
                # Space in the gun view: what was fired -- the dial's last reading (from before the
                # shot, so what it was fired on unless it moved unread) and the tape only if fresh
                self.shot = (self.dial, self.dial_time, self.bearing if fresh_b else None)
                self.fired_since_f8 = True
        if self.k_gun.pressed():
            at, r = self.reader.read()
            if r.ok:
                self.state["gun"] = [r.x, r.y]
                self.state["gun_time"] = time.time()
                self.adj = None
                self.hull_heading, self.hull_proven = None, False
                self.gun_origin, self.gun_source = [r.x, r.y], "F2"
                self.state["f2_gun"], self.state["f2_time"] = [r.x, r.y], self.state["gun_time"]
                self.minimap_gen += 1              # a read already under way is from before this
                if self.state.get("map"):
                    src, (sx, sy) = map_frame(self.state["map"])
                    self.minimap_seed = (src, r.x - sx / 100, r.y - sy / 100)   # the minimap tracks from here
                save_state(self.state)
                self.solve()
                self.render(f"gun stored at x{r.x:.2f} y{r.y:.2f}")
                print(f"gun x{r.x:.2f} y{r.y:.2f}", flush=True)
                beep(1400, 60)
            else:
                path = self.reader.save_debug(at, r.why)
                self.render(f"could not read the map there: {r.why}")
                print(f"F2: {r.why} -- saved {path}", flush=True)
                beep(400, 150)

        if self.k_capture.pressed():
            self.capture_minimap()
        if self.k_impact.pressed():
            self.mark_impact()

        self.k_aim.pressed()
        if self.k_aim.was_down:
            now = time.perf_counter()
            if self.aim_since is None:
                self.aim_since, self.aim_read_ok, self.aim_reported = now, False, False
                self.solve()
                self.render()
            at, r = self.reader.read()
            if r.ok:
                self.aim_read_ok = True
                if not self.state.get("gun"):
                    self.target = (r.x, r.y)
                    self.render("store the gun first: F2 with the cursor on it")
                    return
                if (r.x, r.y) != self.target:
                    self.target = (r.x, r.y)
                    self.solve()
                    self.render("")
            elif not self.aim_read_ok and not self.aim_reported and now - self.aim_since >= F6_FAIL_S:
                self.aim_reported = True
                path = self.reader.save_debug(at, r.why)
                self.render(f"could not read the map there: {r.why}")
                print(f"F3: {r.why} -- saved {path}", flush=True)
                beep(400, 150)
        elif self.aim_since is not None:
            self.aim_since = None
            if self.solution is not None:
                s = self.solution
                print(f"target x{self.target[0]:.2f} y{self.target[1]:.2f}: "
                      f"{s.distance_m:.0f} m, az {s.azimuth_deg:.1f}, "
                      + ", ".join(f"{arc} {ballistics.format_mil(m)}" for arc, m in s.mils.items()),
                      flush=True)


def add_arguments(ap):
    """The artillery options."""
    ap.add_argument("--map", choices=MAP_CHOICES,
                    help="artillery: which map's terrain grid to show the height difference from "
                         "(also the window's map button); default: the last one picked")
    ap.add_argument("--gun-key", type=vk_code, default="F2",
                    help="artillery: store the gun by hand, cursor on it on the map (default F2; the "
                         "minimap places it too)")
    ap.add_argument("--aim-key", type=vk_code, default="F3",
                    help="artillery: hold to show the solution to the cursor (default F3)")
    ap.add_argument("--capture-key", type=vk_code, default="F5",
                    help="artillery: save the screen (minimap) with the gun F2 stored, for minimap "
                         "matching (default F5)")
    ap.add_argument("--impact-key", type=vk_code, default="F4",
                    help="artillery: mark where the shell landed (default F4)")
    ap.add_argument("--fire-key", type=vk_code, default="SPACE",
                    help="artillery: the key you fire with; the dial is noted as it goes (default SPACE)")
    ap.add_argument("--record-bearing", action="store_true",
                    help="artillery: save every bearing read to test_shots/bearing_live/<time>/ for "
                         "hud_bearing.py --replay, and every MIL scale read to its mil/ folder")
    ap.add_argument("--no-tilt", action="store_true",
                    help="artillery: don't read the SPH-2's tilt dial or correct the mils for it")
    ap.add_argument("--window-x", type=int,
                    help="artillery window position (default: where it was last closed, else the "
                         "side screen)")
    ap.add_argument("--window-y", type=int)


def open_window(args, standalone=True):
    """Create the window and its App, placed per args. Call dpi_aware() first."""
    if not os.path.exists(map_coords.GLYPHS):
        n, _ = map_coords.build()
        print(f"built {map_coords.GLYPHS} from {n} digits", flush=True)
    root = tk.Tk()
    app = App(root, args, standalone)
    x, y = window_position(args, app.state, app.reader.sct.monitors)
    root.geometry(f"+{x}+{y}")
    return app


def window_position(args, state, monitors):
    """Where the window opens: --window-x/-y if given, else where it was last closed while
    that spot is still on a screen, else 40 px into the side screen (off the game's)."""
    if args.window_x is not None and args.window_y is not None:
        return args.window_x, args.window_y
    saved = state.get("window")
    if saved and any(m["left"] <= saved[0] < m["left"] + m["width"]
                     and m["top"] <= saved[1] < m["top"] + m["height"] for m in monitors[1:]):
        return saved[0], saved[1]
    m = side_monitor(monitors)
    return m["left"] + 40, m["top"] + 40


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(ap)
    args = ap.parse_args(argv)
    dpi_aware()
    app = open_window(args)
    print("F2: gun at cursor   F3 (hold): solution to cursor   F4: impact at cursor   "
          "close the window to quit", flush=True)
    app.root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
