"""artillery.py: the shot log, the window and the grid heights. Run: python -m pytest"""


def read_rows(path):
    """Every row of a CSV file, the file closed again."""
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def test_shot_log_writes_the_header_once(tmp_path):
    import artillery
    path = tmp_path / "shots.csv"
    artillery.log_shot({"time": "t1", "long_m": 12.5, "right_m": -3}, str(path))
    artillery.log_shot({"time": "t2", "long_m": -7, "notes": "barrel sideways"}, str(path))
    rows = read_rows(path)
    assert [r["time"] for r in rows] == ["t1", "t2"]
    assert rows[0]["long_m"] == "12.5" and rows[1]["notes"] == "barrel sideways"
    assert list(rows[0]) == artillery.SHOT_FIELDS


def test_shot_log_from_an_older_version_gets_the_new_columns(tmp_path):
    import csv
    import artillery
    path = tmp_path / "shots.csv"
    old = [f for f in artillery.SHOT_FIELDS if f != "mil_low_height"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow({"time": "old", "long_m": -203.6, "mil_low": 235.6})
    artillery.log_shot({"time": "new", "mil_low_height": 229.5}, str(path))
    rows = read_rows(path)
    assert list(rows[0]) == artillery.SHOT_FIELDS
    assert (rows[0]["time"], rows[0]["long_m"], rows[0]["mil_low"]) == ("old", "-203.6", "235.6")
    assert (rows[1]["time"], rows[1]["mil_low_height"]) == ("new", "229.5")


class _Key:
    def __init__(self):
        self.queue, self.was_down = [], False

    def pressed(self):
        return self.queue.pop(0) if self.queue else False


def _app(tmp_path, monkeypatch, hud_value=100, tilt=None, no_tilt=False, dial=None):
    import tkinter as tk
    import types
    import artillery
    import map_coords
    monkeypatch.setattr(artillery, "STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(artillery, "SHOTS", str(tmp_path / "shots.csv"))
    monkeypatch.setattr(artillery, "TILT_EVERY_S", 0.0)     # every step reads the dial
    monkeypatch.setattr(artillery, "AZ_BIAS_DEG", 0.0)  # the tests below aim on the target; the bias has its own

    class Reader:
        map = map_coords.Reading(85.95, 54.73)
        tilts = [tilt]          # what successive dial reads return (the last one repeats)
        tilt_calls = 0

        def read(self):
            return (0, 0), self.map

        def read_tilt(self):
            self.tilt_calls += 1
            t = self.tilts.pop(0) if len(self.tilts) > 1 else self.tilts[0]
            return (t, "") if t is not None else (None, "no tilt dial (map open, or not in the vehicle?)")

        bearing = None              # the true tape bearing
        evidence = True             # compass evidence leaning to the true step (None: no compass)

        def read_bearing(self):
            import hud_bearing
            b = self.bearing
            if b is None:
                return None, "no bearing tape"
            k = round(b / 15)
            ev = None
            if self.evidence:
                ev = [1.5] * 24
                ev[k % 24] += 0.6
            return hud_bearing.TapeObs(b - 15 * k, None, ev), ""

        def read_mil(self):
            import hud_mil
            d = self.dial
            return (hud_mil.Mil(d, 2), "") if d is not None else (None, "no MIL scale (map open, ...)")

        def save_debug(self, at, why):
            return ""

        def grab_game(self):
            import numpy as np
            return np.zeros((1440, 2560, 3), np.uint8)

        def grab_minimap(self):
            import numpy as np
            return np.zeros((306, 341, 3), np.uint8)

        def grab_ribbon(self):
            import numpy as np
            return np.full((36, 840, 3), 200, np.uint8)

    for attempt in range(3):            # Tk now and then cannot read its own library files here
        try:                            # ("couldn't read file .../button.tcl"): an environment flake
            root = tk.Tk()
            break
        except tk.TclError:
            if attempt == 2:
                raise
    root.withdraw()
    args = types.SimpleNamespace(gun_key=0, aim_key=0, impact_key=0,
                                 window_x=0, window_y=0, no_tilt=no_tilt)
    app = artillery.App(root, args, standalone=False)
    app.reader = Reader()
    app.locator = _Locator()        # no real map caches in these tests: the minimap never places
    app.minimap_pool = None         # ...and it is matched inline, not on the worker thread
    app.reader.dial = dial
    app.k_gun, app.k_aim, app.k_impact = _Key(), _Key(), _Key()
    app.k_fire = _Key()
    return app, root


def test_f5_stores_the_gun_and_reads_no_asl(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.k_gun.queue.append(True)
        app.step()
        assert app.state["gun"] == [85.95, 54.73]
        assert "gun_asl" not in app.state and "ASL" not in app.vars["gun"].get()
    finally:
        root.destroy()


def test_dz_is_the_grid_difference_and_feeds_both_arcs(tmp_path, monkeypatch):
    import terrain
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.state["gun"], app.target = [85.95, 54.73], (102.35, 62.87)
        app.solve()
        assert app.dz() is None and app.solution.height == {}     # no map: level ground assumed
        monkeypatch.setattr(terrain, "grid_dz", lambda m, g, t, directory=None: 30.0)
        app.state["map"] = "bakurani"
        app.solve()
        assert app.dz() == 30.0
        assert set(app.solution.height) == {"low", "high"}
        flat = app.solution.mils["high"].mil
        mil, why = app.solution.height["high"]
        assert mil is not None and mil < flat                     # a higher target wants a smaller mil
    finally:
        root.destroy()


# --- the tilt dial ----------------------------------------------------------------
ONE_DEG = 17.453          # mil in one degree
GUN, TARGET = (85.95, 54.73), (102.35, 62.87)     # 1831 m: both SPH-2 arcs have a mil


def _nose_up(deg, roll=0.0, turret=0.0, stabilized=True):
    import hud_tilt
    return hud_tilt.Tilt(deg / hud_tilt.DEG_PER_TICK, roll, turret, stabilized)


def ballistics_round(v):
    import ballistics
    return ballistics.js_round(v)


def _aimed(app, target=TARGET):
    """F2 on the gun (the fake map reads GUN), then a target, solved and shown."""
    app.k_gun.queue.append(True)
    app.step()
    app.target = target
    app.solve()
    app.render()
    return app.solution


def _settle(app):
    """Step until the bearing tracker has settled: the turret nudged 1.2 deg back and forth so
    each read is a new view (hud_bearing.NEW_VIEW_DEG), then back on the bearing set."""
    import hud_bearing
    base = app.reader.bearing
    for i in range(max(hud_bearing.CONFIRM_VIEWS, hud_bearing.SETTLE_READS) + 2):
        app.reader.bearing = base + (1.2 if i % 2 else 0.0)
        app.step()
    app.reader.bearing = base
    app.step()


def test_the_dial_moves_both_spg_arcs_by_the_tilt_along_the_gun(tmp_path, monkeypatch):
    import pytest
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(1.0))
    try:
        s = _aimed(app)
        # nose up 1 deg with the gun forward points the barrel 1 deg higher on either arc
        assert app.shown_mil(s, "low") == pytest.approx(s.mils["low"].mil - ONE_DEG, abs=0.01)
        assert app.shown_mil(s, "high") == pytest.approx(s.mils["high"].mil - ONE_DEG, abs=0.01)
        assert "tilt" not in app.vars["mils"].get()          # the green holds the mils only
        assert str(ballistics_round(app.shown_mil(s, "low"))) in app.vars["mils"].get()
        assert "nose up 1.0" in app.vars["tilt"].get() and "Stabilized" in app.vars["tilt"].get()
    finally:
        root.destroy()


def test_a_gun_square_to_the_hull_takes_the_roll_not_the_pitch(tmp_path, monkeypatch):
    import pytest
    # nose up 4 deg, right side up 1 deg, gun to the right: the low arc (~9 deg) rises ~1 deg
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(4.0, roll=1.0, turret=90.0))
    try:
        s = _aimed(app)
        assert app.shown_mil(s, "low") == pytest.approx(s.mils["low"].mil - ONE_DEG, abs=1.0)
    finally:
        root.destroy()


def test_no_tilt_neither_reads_the_dial_nor_corrects(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(3.0), no_tilt=True)
    try:
        s = _aimed(app)
        assert app.reader.tilt_calls == 0
        assert app.shown_mil(s, "low") == s.mils["low"].mil
        assert "off" in app.vars["tilt"].get()
    finally:
        root.destroy()


def test_a_failed_dial_read_keeps_the_last_good_one(tmp_path, monkeypatch):
    import pytest
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(2.0))
    try:
        app.reader.tilts = [_nose_up(2.0), None]            # the map opens after the first read
        app.step()
        app.step()
        assert app.reader.tilt_calls == 2
        s = _aimed(app)
        assert app.shown_mil(s, "low") == pytest.approx(s.mils["low"].mil - 2 * ONE_DEG, abs=0.01)
    finally:
        root.destroy()


def test_f8_logs_the_tilt_the_shot_was_fired_with(tmp_path, monkeypatch):
    import map_coords
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(1.0, roll=-2.5, turret=30.0, stabilized=False))
    try:
        _aimed(app)
        app.reader.map = map_coords.Reading(102.30, 62.80)    # the impact
        app.k_impact.queue.append(True)
        app.step()
        row = read_rows(tmp_path / "shots.csv")[-1]
        assert (row["tilt_pitch_deg"], row["tilt_roll_deg"], row["tilt_turret_deg"]) == ("1.0", "-2.5", "30.0")
        assert row["tilt_stabilized"] == "no" and row["tilt_age_s"] == "0"
        # 1 deg x cos 30 - 2.5 deg x sin 30 = -0.384 deg along the gun: the dial goes UP ~6.7 mil
        assert abs(float(row["tilt_low"]) - 6.7) <= 0.3
    finally:
        root.destroy()


def test_f8_twice_on_the_same_impact_corrects_once(tmp_path, monkeypatch):
    # Live 2026-09-29 21:51: ten F8s on one impact logged ten rows and stacked ten
    # corrections, walking the high mil from 1062 to 700 for a single 94 m miss.
    import map_coords
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        app.reader.map = map_coords.Reading(102.30, 62.80)
        app.k_impact.queue.append(True)
        app.step()
        once = app.shown_mil(s, "high")
        app.reader.map = map_coords.Reading(102.31, 62.79)    # the same crater, a pixel off
        app.k_impact.queue.append(True)
        app.step()
        assert app.shown_mil(s, "high") == once
        assert len(read_rows(tmp_path / "shots.csv")) == 1
        app.last_impact = (app.last_impact[0], app.last_impact[1], app.last_impact[2] - 9)
        app.reader.dial = app.shown_mil(s, "high")
        app.step()
        app.k_fire.queue.append(True)                         # fired again (Space) ...
        app.step()
        app.reader.map = map_coords.Reading(102.31, 62.79)    # ... into the same crater, 9 s on
        app.k_impact.queue.append(True)
        app.step()
        assert len(read_rows(tmp_path / "shots.csv")) == 2
        app.reader.map = map_coords.Reading(102.00, 62.50)    # a new shot somewhere else stacks
        app.k_impact.queue.append(True)
        app.step()
        assert app.shown_mil(s, "high") != once
    finally:
        root.destroy()


# --- the MIL scale ------------------------------------------------------------------
def _fire(app, impact):
    import map_coords
    app.reader.map = map_coords.Reading(*impact)
    app.k_impact.queue.append(True)
    app.step()
    return read_rows(app_shots(app))[-1]


def app_shots(app):
    import artillery
    return artillery.SHOTS


def test_the_window_says_how_far_to_turn_the_dial(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        high = app.shown_mil(s, "high")
        app.reader.dial = high - 4.3
        app.step()
        text = app.vars["dial"].get()
        assert "HIGH" in text and "+4.3" in text and "LOW" in text
        assert text.index("LOW") < text.index("HIGH")                  # low first, one arc per line
        assert f"{high - 4.3:.1f}" not in text                         # the remaining turn, not the dial
    finally:
        root.destroy()


def test_f8_logs_the_mil_the_dial_showed(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        app.reader.dial = app.shown_mil(s, "high") + 2.0
        app.step()
        app.reader.dial = None                  # the map is open for the F4
        row = _fire(app, (102.30, 62.80))
        assert row["fired_arc"] == "high" and row["dial_age_s"] == "0"
        assert abs(float(row["dial_mil"]) - float(row["shown_high"]) - 2.0) <= 0.11
    finally:
        root.destroy()


def test_a_hit_fired_off_the_shown_mil_keeps_what_was_fired(tmp_path, monkeypatch):
    # the shell landed ON the target, but the dial was 5 mil off what the window
    # showed: the next shot wants what was really fired, not the shown mil again
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        fired = app.shown_mil(s, "high") + 5.0
        app.reader.dial = fired
        app.step()
        _fire(app, TARGET)
        assert app.shown_mil(s, "high") == pytest_approx(fired)
    finally:
        root.destroy()


def test_a_dial_far_from_every_arc_is_not_taken_as_fired(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        shown = app.shown_mil(s, "high")
        app.reader.dial = shown + 60            # still turning, or another target's mil
        app.step()
        row = _fire(app, TARGET)
        assert row["fired_arc"] == "" and app.shown_mil(s, "high") == pytest_approx(shown)
    finally:
        root.destroy()


def pytest_approx(v):
    import pytest
    return pytest.approx(v, abs=0.05)


def test_space_keeps_the_dial_it_was_fired_on(tmp_path, monkeypatch):
    # fired on 1300.4, then the dial turned toward the next shot before the F4
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        fired = app.shown_mil(s, "high") + 1.0
        app.reader.dial = fired
        app.k_fire.queue.append(True)
        app.step()
        app.reader.dial = fired - 12.0
        app.step()
        row = _fire(app, (102.30, 62.80))
        assert abs(float(row["dial_mil"]) - fired) < 0.01 and row["fired_by"] == "space"
    finally:
        root.destroy()


def test_space_without_the_gun_view_is_not_a_shot(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        fired = app.shown_mil(s, "high")
        app.reader.dial = fired
        app.k_fire.queue.append(True)
        app.step()                               # the shot
        app.reader.dial = None                   # on foot, map open: no fresh dial
        app.dial_time -= 5
        app.k_fire.queue.append(True)
        app.step()                               # Space again (a jump), not a shot
        row = _fire(app, (102.30, 62.80))
        assert abs(float(row["dial_mil"]) - fired) < 0.01 and row["fired_by"] == "space"
    finally:
        root.destroy()


def test_without_a_space_the_last_dial_reading_is_logged(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        app.reader.dial = app.shown_mil(s, "high")
        app.step()
        row = _fire(app, (102.30, 62.80))
        assert row["fired_by"] == "last reading"
    finally:
        root.destroy()


# --- the bearing tape ---------------------------------------------------------------
def test_the_window_settles_then_says_how_far_to_traverse(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        app.reader.bearing = (s.azimuth_deg + 1.3) % 360
        app.step()
        assert "settling" in app.vars["bearing"].get()       # no number before it is sure
        _settle(app)
        text = app.vars["bearing"].get()
        assert "-1.3" in text and f"{(s.azimuth_deg + 1.3) % 360:.1f}" not in text
    finally:
        root.destroy()


def test_f8_logs_the_bearing_fired_on_and_corrects_from_it(tmp_path, monkeypatch):
    # fired 1.3 deg right of the target and landed 1.3 deg right: the next shot wants
    # the target's own azimuth again, not the shown one minus the miss
    import math
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        app.reader.dial = app.shown_mil(s, "high")
        app.reader.bearing = s.azimuth_deg + 1.3
        _settle(app)
        app.k_fire.queue.append(True)
        app.step()
        app.reader.bearing = s.azimuth_deg - 4        # turning away before the F4
        app.step()
        g = app.state["gun"]
        a = math.radians(s.azimuth_deg + 1.3)
        impact = (g[0] + math.sin(a) * s.distance_m / 100, g[1] + math.cos(a) * s.distance_m / 100)
        row = _fire(app, impact)
        assert float(row["bearing_deg"]) == pytest_approx(s.azimuth_deg + 1.3)
        assert app.adjustment()["az"] == pytest_approx(0.0)
    finally:
        root.destroy()


def test_space_with_the_mil_scale_out_still_keeps_the_bearing_fired_on(tmp_path, monkeypatch):
    # Live 2026-10-13 16:55: the MIL scale had just dropped out, Space went unrecorded, and F4
    # corrected from the tape after the turret turned (84.7 for a shot fired on 88.9)
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        app.reader.dial = app.shown_mil(s, "high")
        app.reader.bearing = s.azimuth_deg + 1.3
        _settle(app)
        app.reader.dial = None                        # the MIL scale drops out...
        app.dial_time -= 5
        app.k_fire.queue.append(True)                 # ...and Space fires on the tape alone
        app.step()
        app.reader.bearing = s.azimuth_deg - 4        # turning away before the F4
        _settle(app)
        row = _fire(app, (102.30, 62.80))
        assert float(row["bearing_deg"]) == pytest_approx(s.azimuth_deg + 1.3)
        assert row["fired_by"] == "space"             # on the dial's last reading, from before the shot
    finally:
        root.destroy()


def test_a_shot_fired_unseen_is_not_corrected_from_a_later_bearing(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        app.reader.dial = app.shown_mil(s, "high")
        app.reader.bearing = s.azimuth_deg - 4        # where the turret is at the F4, not where it fired
        _settle(app)
        row = _fire(app, (102.30, 62.80))
        assert row["bearing_deg"] == "" and row["barrel_az_deg"] == ""
    finally:
        root.destroy()


def test_every_shot_is_aimed_the_bias_right_and_f4_learns_on_top_of_it(tmp_path, monkeypatch):
    # shells land ~1 deg left of the barrel (41 shots, 3 maps): the first shot aims 1 deg right;
    # one that lands only 0.4 deg left of it (0.6 right of the target) aims the next 0.4 right
    import math
    import artillery
    app, root = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(artillery, "AZ_BIAS_DEG", 1.0)
    try:
        s = _aimed(app)
        assert app.aim_az(s) == pytest_approx(s.azimuth_deg + 1.0)
        assert app.bearing_goal(s) % 360 == pytest_approx((s.azimuth_deg + 1.0) % 360)
        app.reader.dial = app.shown_mil(s, "high")
        app.reader.bearing = (s.azimuth_deg + 1.0) % 360
        _settle(app)
        app.k_fire.queue.append(True)
        app.step()
        g = app.state["gun"]
        a = math.radians(s.azimuth_deg + 0.6)                  # landed 0.4 deg left of the barrel
        impact = (g[0] + math.sin(a) * s.distance_m / 100, g[1] + math.cos(a) * s.distance_m / 100)
        row = _fire(app, impact)
        assert float(row["shown_az"]) == pytest_approx((s.azimuth_deg + 1.0) % 360)
        assert app.aim_az(s) == pytest_approx(s.azimuth_deg + 0.4)    # the barrel where that shell hits
    finally:
        root.destroy()


def test_the_same_crater_marked_again_without_a_shot_is_not_counted(tmp_path, monkeypatch):
    # Live 2026-09-30 11:52: one crater F4'd twice 23 s apart stacked a second correction
    import map_coords
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        app.reader.map = map_coords.Reading(102.30, 62.80)
        app.k_impact.queue.append(True)
        app.step()
        once = app.shown_mil(s, "high")
        app.last_impact = (app.last_impact[0], app.last_impact[1], app.last_impact[2] - 23)
        app.k_impact.queue.append(True)                       # no Space in between
        app.step()
        assert app.shown_mil(s, "high") == once
        assert len(read_rows(tmp_path / "shots.csv")) == 1
    finally:
        root.destroy()


def test_the_next_shot_aims_at_the_mean_of_every_shot_so_far(tmp_path, monkeypatch):
    # each shot says where the dial should have been (what it fired + its miss); a single
    # shot scatters ~8 m, so the next aim is their mean, not the last one alone
    import math
    import ballistics
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        w = app.weapons["spg"]
        g = app.state["gun"]
        ests = []
        for dial_off, long_m in ((0.0, -26.0), (-6.0, 15.0), (-3.0, 2.0)):
            fired = app.shown_mil(s, "high") + dial_off
            app.reader.dial = fired
            app.step()
            app.k_fire.queue.append(True)
            app.step()
            a = math.radians(s.azimuth_deg)
            d = (s.distance_m + long_m) / 100
            ests.append(fired + ballistics.adjust(w, "high", s.distance_m, None, long_m))
            _fire(app, (g[0] + math.sin(a) * d, g[1] + math.cos(a) * d))
        assert app.shown_mil(s, "high") == pytest_approx(sum(ests) / 3)
    finally:
        root.destroy()


def test_the_grid_line_waits_for_a_map_then_shows_the_terrain_difference(tmp_path, monkeypatch):
    import terrain
    app, root = _app(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(terrain, "grid_dz", lambda m, g, t, directory=None: 24.4 if m == "bakurani" else None)
        app.state["gun"], app.target = [85.95, 54.73], (102.35, 62.87)
        app.render()
        assert "pick the map" in app.vars["dz"].get() and "not picked" in app.map_label.get()
        app.cycle_map()
        assert app.state["map"] == "bakurani"
        assert "+24 m (target higher)" in app.vars["dz"].get()
        app.cycle_map()
        assert app.state["map"] == "ozeti" and "no ozeti terrain data" in app.vars["dz"].get()
        app.cycle_map(); app.cycle_map()
        assert app.state["map"] == "bakurani-training"
        app.cycle_map()
        assert app.state["map"] is None                       # the button cycles back to "not picked"
    finally:
        root.destroy()


def test_f8_logs_the_map_and_the_grid_difference(tmp_path, monkeypatch):
    import terrain
    app, root = _app(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(terrain, "grid_dz", lambda m, g, t, directory=None: -12.34)
        app.state["gun"], app.state["map"], app.target = [85.95, 54.73], "zestafona", (102.35, 62.87)
        app.solve()
        app.reader.map = __import__("map_coords").Reading(102.5, 63.0)
        app.mark_impact()
        row = read_rows(tmp_path / "shots.csv")[0]
        assert row["map"] == "zestafona" and row["grid_dz_m"] == "-12.3"
    finally:
        root.destroy()


def test_dz_reads_the_grid_at_the_measured_map_offset(tmp_path, monkeypatch):
    import pytest
    import terrain
    seen = []
    monkeypatch.setattr(terrain, "grid_dz", lambda m, g, t, directory=None: seen.append((m, g, t)) or 5.0)
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.state["gun"], app.target = [80.0, 30.0], (90.0, 40.0)
        app.state["map"] = "bakurani"
        assert app.dz() == 5.0
        m, g, t = seen[-1]
        assert g == pytest.approx((79.72, 29.86)) and t == pytest.approx((89.72, 39.86))     # (-28, -14) m
        app.state["map"] = "ozeti"                                   # not measured: no shift
        app.dz()
        assert seen[-1][1] == (80.0, 30.0)
    finally:
        root.destroy()


def test_the_traverse_line_turns_the_tape_off_the_azimuth_for_a_rolled_hull(tmp_path, monkeypatch):
    import pytest
    import ballistics
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, roll=12.6, turret=0.0))
    try:
        s = _aimed(app)
        app.reader.bearing = s.azimuth_deg                       # tape on the plain azimuth
        app.reader.dial = app.shown_mil(s, "low")                # the arc being set
        _settle(app)
        goal = app.bearing_goal(s)
        arc_mil = app.world_mil(s, "low")
        turret, _ = ballistics.aim(s.azimuth_deg - 0.0, 0.0, 12.6, s.azimuth_deg, arc_mil, turret0=0.0)
        assert goal == pytest.approx((s.azimuth_deg + turret) % 360, abs=0.05)
        assert abs(goal - s.azimuth_deg) > 1.0                   # a real swing at ~9 deg of barrel
        # a level hull needs no swing
        app2, root2 = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, roll=0.0, turret=0.0))
        try:
            s2 = _aimed(app2)
            app2.reader.bearing = s2.azimuth_deg
            _settle(app2)
            assert app2.bearing_goal(s2) == pytest.approx(s2.azimuth_deg, abs=0.05)
        finally:
            root2.destroy()
    finally:
        root.destroy()


def test_f8_logs_where_the_rolled_barrel_really_pointed(tmp_path, monkeypatch):
    import pytest
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, roll=12.6, turret=0.0))
    try:
        s = _aimed(app)
        app.reader.bearing = s.azimuth_deg
        app.reader.dial = app.shown_mil(s, "low")
        _settle(app)
        app.k_fire.queue.append(True)
        app.step()
        app.reader.dial, app.reader.bearing = None, None             # map open for the F4
        row = _fire(app, (102.30, 62.80))
        assert float(row["bearing_deg"]) == pytest.approx(s.azimuth_deg, abs=0.05)
        assert float(row["barrel_az_deg"]) < float(row["bearing_deg"]) - 0.5    # swung left of the tape
    finally:
        root.destroy()


def test_a_gap_after_a_traverse_settles_then_reads(tmp_path, monkeypatch):
    # the map was open while the turret turned 40 deg: the anchored hull heading plus the
    # turret angle (trusted) gives the step back at once
    import pytest
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=0.0))
    try:
        s = _aimed(app)
        hull = round(s.azimuth_deg)
        app.reader.bearing = float(hull)
        _settle(app)
        assert app.bearing == pytest.approx(hull, abs=0.01) and app.hull_heading is not None
        app.reader.tilts = [_nose_up(0.0, turret=40.0)]
        app.reader.bearing = float(hull + 40)
        app.tracker.t -= 5.0                               # a 5 s gap without reads: everything
        app.bearing_time -= 5.0                            # read before it is 5 s old
        app.tilt_time -= 5.0
        app.turret_hist = [(t - 5.0, tu, tp) for t, tu, tp in app.turret_hist]
        for _ in range(hud_bearing_settle_reads()):          # trusted: a brief, uncontradicted settle
            app.step()
        assert app.bearing == pytest.approx(hull + 40, abs=0.01)
    finally:
        root.destroy()


def hud_bearing_settle_reads():
    import hud_bearing
    return hud_bearing.SETTLE_READS


def test_a_guess_never_becomes_the_hull_anchor(tmp_path, monkeypatch):
    # review 2026-10-11: with no compass evidence the target's azimuth used to be shown after a
    # few reads and then anchored as the hull heading -- a guess trusted from then on
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=0.0))
    try:
        s = _aimed(app)
        app.reader.evidence = None
        app.reader.bearing = (s.azimuth_deg + 12.0) % 360
        for _ in range(40):
            app.step()
        assert app.bearing is None and app.hull_heading is None
        app.tracker.since -= 3.0                          # settling 3 s: ask for a little traverse
        app.step()
        assert "turn the turret" in app.vars["bearing"].get()
    finally:
        root.destroy()


def test_a_switch_reanchors_the_hull_at_once(tmp_path, monkeypatch):
    # review 2026-10-11: a corrected step left the hull heading wrong for 8 reads, and a map
    # opened in that time brought the wrong number back as trusted
    import pytest
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=0.0))
    try:
        s = _aimed(app)
        truth = round(s.azimuth_deg) * 1.0
        app.reader.bearing = truth
        app.step()                                        # the tilt is read: fresh
        app.hull_heading = (truth + 15.0) % 360          # anchored a step wrong
        app.hull_attitude = (0.0, 0.0)
        app.tracker.t = None                              # a fresh start from that anchor
        for i in range(60):
            app.reader.bearing = truth + (1.2 if i % 2 else 0.0)
            app.step()
            if app.bearing is not None:
                break
        assert app.bearing == pytest.approx(app.reader.bearing, abs=0.01)
        assert (app.hull_heading - app.reader.bearing + 180) % 360 - 180 == pytest.approx(0.0, abs=0.01)
    finally:
        root.destroy()


def test_a_moved_hull_drops_the_anchor_and_settles(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=0.0))
    try:
        s = _aimed(app)
        app.reader.bearing = round(s.azimuth_deg) * 1.0
        _settle(app)
        app.reader.tilts = [_nose_up(0.0, roll=4.0, turret=0.0)]   # it sits differently: driven
        app.tracker.t -= 5.0
        app.step()
        assert app.hull_heading is None and app.bearing is None
        assert "settling" in app.vars["bearing"].get()
    finally:
        root.destroy()


MONITORS = [{"left": 0, "top": 0, "width": 4480, "height": 1440},
            {"left": 2560, "top": 0, "width": 1920, "height": 1080},
            {"left": 0, "top": 0, "width": 2560, "height": 1440}]


def test_the_window_reopens_where_it_was():
    import types
    import artillery
    no_args = types.SimpleNamespace(window_x=None, window_y=None)
    assert artillery.window_position(no_args, {"window": [3100, 500]}, MONITORS) == (3100, 500)
    # never saved: 40 px into the side screen, off the game
    assert artillery.window_position(no_args, {}, MONITORS) == (2600, 40)
    # saved on a screen that is not there any more: back to the side screen, not lost off-screen
    assert artillery.window_position(no_args, {"window": [6000, 500]}, MONITORS) == (2600, 40)
    # --window-x/-y still win
    given = types.SimpleNamespace(window_x=10, window_y=20)
    assert artillery.window_position(given, {"window": [3100, 500]}, MONITORS) == (10, 20)


def test_closing_the_window_saves_where_it_is(tmp_path, monkeypatch):
    import json
    import artillery
    app, root = _app(tmp_path, monkeypatch)
    try:
        root.winfo_x, root.winfo_y = (lambda: 3100), (lambda: 500)
        app.save_position()
        with open(artillery.STATE, encoding="utf-8") as f:
            assert json.load(f)["window"] == [3100, 500]
    finally:
        root.destroy()


def test_an_f4_correction_carries_to_other_targets_until_the_gun_is_stored_again(tmp_path, monkeypatch):
    # Live 2026-09-30, one gun spot on Bakurani: the first shot at each of six new targets
    # landed 31-65 m long (mean 43) -- the gun's own error, thrown away at every new target.
    import math
    import map_coords
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        g = app.state["gun"]
        a = math.radians(s.azimuth_deg)
        far = (s.distance_m + 40) / 100
        _fire(app, (g[0] + math.sin(a) * far, g[1] + math.cos(a) * far))     # 40 m long
        adj = app.adjustment()
        assert adj and adj["low"] != 0
        app.target = (TARGET[0] - 5.0, TARGET[1] + 2.0)        # ~540 m from the first target
        app.solve()
        assert app.adjustment() is adj                        # same gun: still corrected
        corrected = app.world_mil(app.solution, "low")
        app.adj = None
        plain = app.world_mil(app.solution, "low")
        app.adj = adj
        assert corrected - plain == pytest_approx(adj["low"])   # applied at the new target
        app.reader.map = map_coords.Reading(*GUN)
        app.k_gun.queue.append(True)                          # F2: the gun stored again
        app.step()
        assert app.adjustment() is None
    finally:
        root.destroy()


def test_f5_saves_the_minimap_frame_with_the_stored_gun(tmp_path, monkeypatch):
    # USER 2026-10-11: minimap shots with known positions, to match the minimap on the terrain map.
    # The minimap is hidden while the big map is open: F2 there, close it, then F5.
    import artillery
    monkeypatch.setattr(artillery, "MINIMAP_DIR", str(tmp_path / "minimap"))
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.k_capture = _Key()
        app.k_capture.queue.append(True)
        app.step()                                     # no gun stored yet: nothing saved
        assert not (tmp_path / "minimap").exists()
        _aimed(app)                                    # F2 on the gun
        app.k_capture.queue.append(True)
        app.step()
        for th in __import__("threading").enumerate():
            if th.name.startswith("minimap"):
                th.join()
        rows = read_rows(tmp_path / "minimap" / "log.csv")
        assert len(rows) == 1
        assert (float(rows[0]["gun_x"]), float(rows[0]["gun_y"])) == GUN
        assert float(rows[0]["gun_age_s"]) < 5
        assert (tmp_path / "minimap" / rows[0]["file"]).exists()
    finally:
        root.destroy()


def test_recording_saves_every_mil_scale_read(tmp_path, monkeypatch):
    # USER 2026-10-11: record the MIL scale with the bearing, so a wrong Dial line can be re-read
    import json
    import numpy as np
    import hud_bearing
    app, root = _app(tmp_path, monkeypatch, dial=612.5)
    try:
        app.mil_recorder = hud_bearing.Recorder(str(tmp_path / "mil"))
        app.reader.last_mil = np.zeros((720, 200, 3), np.uint8)
        app.poll_dial()
        assert (tmp_path / "mil" / "00000_strip.png").exists()
        entry = json.loads((tmp_path / "mil" / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert entry["mil"] == 612.5 and entry["i"] == 0
    finally:
        root.destroy()


# --- the minimap (2026-10-12) -----------------------------------------------------------------
class _Locator:
    """A scripted minimap.Locator: locate() hands out the queued (Fix, why) results in turn."""

    def __init__(self, results=()):
        self.results, self.seeds = list(results), []

    delay = 0.0

    def locate(self, crop, now):
        if self.delay:
            __import__("time").sleep(self.delay)
        return self.results.pop(0) if self.results else (None, "minimap not placed (no confident match)")

    def seed(self, map_id, x, y, now):
        self.seeds.append((map_id, x, y))

    only_map = None


def _fix(x, y, heading=90.0, map_id="zestafona"):
    import minimap
    return minimap.Fix(map_id, x, y, heading, 0.9, 0.6), ""


def test_a_minimap_fix_is_the_gun_and_the_map(tmp_path, monkeypatch):
    import pytest
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.locator = _Locator([_fix(65.70, 69.79)])
        app.step()
        assert app.state["gun"] == pytest.approx([65.70, 69.79])
        assert app.state["map"] == "zestafona"
        assert "minimap" in app.vars["gun"].get() and "minimap" in app.map_label.get()
    finally:
        root.destroy()


def test_jitter_keeps_the_f4_correction_and_a_real_move_clears_it(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.locator = _Locator([_fix(65.70, 69.79)])
        app.step()
        app.adj = {"target": (70.0, 70.0), "az": 1.0, "low": -5.0}
        app.locator.results = [_fix(65.72, 69.80)]           # 2 m: the same spot
        app.step()
        assert app.adj is not None
        app.locator.results = [_fix(65.90, 69.79)]           # 20 m: the gun moved
        app.step()
        assert app.adj is None
    finally:
        root.destroy()


def test_the_icon_heading_is_the_bearings_trusted_prior(tmp_path, monkeypatch):
    import pytest
    import minimap
    monkeypatch.setattr(minimap, "icon_heading", lambda crop: (37.0, ""))   # the icon in the frame
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=10.0))
    try:
        _aimed(app)
        app.locator = _Locator([_fix(*app.state["gun"], heading=37.0, map_id=app.state.get("map") or "zestafona")] * 3)
        seen = []
        real = app.tracker.update
        app.tracker.update = lambda now, frac, ev, prior, trusted=False, proof=None: seen.append((prior, trusted)) or \
            real(now, frac, ev, prior, trusted, proof)
        app.reader.bearing = 47.0
        app.step()
        assert seen and seen[-1][1] is True and seen[-1][0] == pytest.approx(47.0)   # 37 + turret 10
    finally:
        root.destroy()


def test_f2_seeds_the_minimap_and_a_refused_read_keeps_the_gun(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.state["map"] = "ozeti"
        app.locator = _Locator()
        _aimed(app)                                         # F2 on the gun
        before = list(app.state["gun"])
        app.step()                                          # the seed goes with the next read...
        assert app.locator.seeds and app.locator.seeds[-1][0] == "ozeti"
        assert app.state["gun"] == before and "not placed" in app.minimap_why   # ...which is refused
    finally:
        root.destroy()


def test_the_minimap_is_matched_off_the_window_loop(tmp_path, monkeypatch):
    import concurrent.futures
    import pytest
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.minimap_pool = concurrent.futures.ThreadPoolExecutor(1)
        app.locator = _Locator([_fix(65.70, 69.79)])
        app.step()                                   # submitted: nothing taken yet
        assert app.minimap_job is not None
        app.minimap_job.result(timeout=5)
        app.step()                                   # the result is taken on the next poll
        assert app.state["gun"] == pytest.approx([65.70, 69.79])
    finally:
        app.shutdown()
        root.destroy()


def test_the_map_button_locks_the_map_and_keeps_the_f4_correction(tmp_path, monkeypatch):
    # review 2026-10-12: a minimap fix switched the map straight back, and counted as a moved gun
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.locator = _Locator([_fix(65.70, 69.79, map_id="ozeti")])
        app.step()
        app.adj = {"target": (70.0, 70.0), "az": 1.0, "low": -5.0}
        app.cycle_map()                                   # ozeti -> zestafona: locked
        locked = app.state["map"]
        assert locked == "zestafona" and app.adj is not None
        app.locator.results = [_fix(65.70, 69.79, map_id="ozeti")]       # the minimap still says ozeti
        app.step()
        assert app.state["map"] == locked and app.adj is not None
        assert app.locator.only_map == locked
    finally:
        root.destroy()


def test_the_training_map_is_bakurani_shifted_into_its_own_coordinates(tmp_path, monkeypatch):
    # 2026-10-14: on the training range the map readout sits 24 m W, 76 m S of the minimap's place
    import artillery
    import terrain
    calls = []
    monkeypatch.setattr(terrain, "grid_dz", lambda m, g, t, directory=None: calls.append((m, g, t)) or 5.0)
    app, root = _app(tmp_path, monkeypatch)
    try:
        while app.state["map"] != "bakurani-training":
            app.cycle_map()
        app.locator = _Locator([_fix(99.27, 108.93, map_id="bakurani")])
        app.step()
        assert app.locator.only_map == "bakurani"                    # the minimap searches bakurani
        assert app.state["map"] == "bakurani-training"                # and the choice stays
        assert app.state["gun"] == [pytest_approx(99.03), pytest_approx(108.17)]
        app.target = (85.37, 112.77)
        assert app.dz() == 5.0
        m, g, t = calls[-1]
        ox, oy = terrain.MAP_OFFSET_M["bakurani"]
        assert m == "bakurani"                                       # bakurani's terrain, read at the
        assert g[0] == pytest_approx(99.27 + ox / 100, ) and g[1] == pytest_approx(108.93 + oy / 100)
        assert t[0] == pytest_approx(85.37 + 0.24 + ox / 100) and t[1] == pytest_approx(112.77 + 0.76 + oy / 100)
    finally:
        root.destroy()


def test_f5_logs_the_f2_position_not_the_minimaps(tmp_path, monkeypatch):
    # review 2026-10-12: F5 logged the minimap's own fix as the ground truth to check it against
    import artillery
    monkeypatch.setattr(artillery, "MINIMAP_DIR", str(tmp_path / "minimap"))
    app, root = _app(tmp_path, monkeypatch)
    try:
        _aimed(app)                                       # F2: GUN
        app.locator.results = [_fix(GUN[0] + 0.05, GUN[1] + 0.03, map_id="zestafona")]
        app.step()                                        # the minimap moves the gun 6 m
        assert app.state["gun"] != list(GUN)
        app.k_capture = _Key()
        app.k_capture.queue.append(True)
        app.step()
        for th in __import__("threading").enumerate():
            if th.name.startswith("minimap"):
                th.join()
        row = read_rows(tmp_path / "minimap" / "log.csv")[-1]
        assert (float(row["gun_x"]), float(row["gun_y"])) == GUN
        assert row["gun_source"] == "minimap" and float(row["mm_x"]) == pytest_approx(GUN[0] + 0.05)
    finally:
        root.destroy()


def test_a_minimap_error_is_a_reason_not_a_broken_window(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        def boom(crop, now):
            raise ValueError("bad crop")
        app.locator.locate = boom
        app.step()
        assert "bad crop" in app.minimap_why
        app.step()                                        # and the window keeps going
    finally:
        root.destroy()


def test_a_stale_result_after_f2_is_dropped(tmp_path, monkeypatch):
    import concurrent.futures
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.minimap_pool = concurrent.futures.ThreadPoolExecutor(1)
        app.locator = _Locator([_fix(65.70, 69.79)])
        app.locator.delay = 0.3
        app.step()                                        # a read in flight (0.3 s)...
        _aimed(app)                                       # ...F2 meanwhile: GUN
        app.minimap_job.result(timeout=5)
        app.step()                                        # the read from before F2 comes back
        assert app.state["gun"] == list(GUN)
    finally:
        app.shutdown()
        root.destroy()


def test_the_first_fix_keeps_an_f4_correction(tmp_path, monkeypatch):
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.gun_origin = None
        app.adj = {"target": (70.0, 70.0), "az": 1.0, "low": -5.0}
        app.locator = _Locator([_fix(65.70, 69.79)])
        app.step()
        assert app.adj is not None
    finally:
        root.destroy()


def test_closing_the_window_stops_the_minimap_worker(tmp_path, monkeypatch):
    import concurrent.futures
    import pytest
    app, root = _app(tmp_path, monkeypatch)
    try:
        pool = app.minimap_pool = concurrent.futures.ThreadPoolExecutor(1)
        app.shutdown()
        with pytest.raises(RuntimeError):
            pool.submit(lambda: None)
    finally:
        root.destroy()


def test_the_icon_heading_is_used_even_where_the_minimap_cannot_place_us(tmp_path, monkeypatch):
    # Zestafona: the minimap rarely places the tank, but its icon still says where the hull points
    import pytest
    import minimap
    monkeypatch.setattr(minimap, "icon_heading", lambda crop: (37.0, ""))
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=10.0))
    try:
        _aimed(app)
        app.locator = _Locator()                          # never places
        seen = []
        real = app.tracker.update
        app.tracker.update = lambda now, frac, ev, prior, trusted=False, proof=None: seen.append((prior, trusted)) or \
            real(now, frac, ev, prior, trusted, proof)
        app.reader.bearing = 47.0
        app.step()
        assert seen and seen[-1] == (pytest.approx(47.0), True)     # 37 + turret 10
    finally:
        root.destroy()


def test_the_icon_heading_stays_fresh_while_a_slow_minimap_search_runs(tmp_path, monkeypatch):
    # Live 2026-10-13 (overcast Zestafona): whole-map searches took 0.4-0.7 s and the heading came
    # back with them, older than MM_HEADING_FRESH_S -- the tape never got its trusted prior
    import concurrent.futures
    import threading
    import time
    import artillery
    import minimap
    monkeypatch.setattr(minimap, "icon_heading", lambda crop: (37.0, ""))
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=10.0))
    release = threading.Event()

    class _Slow:
        only_map = None

        def seed(self, *a):
            pass

        def locate(self, crop, now):
            release.wait(5)
            return None, "minimap not placed (no confident match: featureless ground?)"
    try:
        app.locator = _Slow()
        app.minimap_pool = concurrent.futures.ThreadPoolExecutor(1)
        app.icon_heading = None
        grabs = []
        real = app.reader.grab_minimap
        app.reader.grab_minimap = lambda: grabs.append(1) or real()
        app.poll_minimap()                                # the search starts and hangs
        app.icon_heading = None
        app.poll_minimap()                                # still searching: too soon to grab again
        assert len(grabs) == 1 and app.icon_heading is None
        time.sleep(artillery.ICON_EVERY_S + 0.05)
        app.poll_minimap()                                # still searching: the icon is read anyway
        assert len(grabs) == 2
        assert app.icon_heading is not None and app.icon_heading[0] == 37.0
        assert time.monotonic() - app.icon_heading[1] < 0.2
    finally:
        release.set()
        app.minimap_pool.shutdown(wait=True)
        app.minimap_pool = None
        root.destroy()


def test_a_recorded_bearing_read_logs_the_minimap_and_the_icon(tmp_path, monkeypatch):
    # 2026-10-12: a renamed variable broke --record-bearing and no test noticed
    import json
    import numpy as np
    import hud_bearing
    import minimap
    monkeypatch.setattr(minimap, "icon_heading", lambda crop: (37.0, ""))
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=10.0))
    try:
        _aimed(app)
        app.recorder = hud_bearing.Recorder(str(tmp_path / "rec"))
        app.reader.last_tape, app.reader.last_box = np.zeros((90, 720, 3), np.uint8), None
        app.reader.last_tilt = np.zeros((250, 280, 3), np.uint8)
        app.reader.bearing = 47.0
        app.step()
        line = json.loads((tmp_path / "rec" / "log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert line["icon_heading"] == 37.0 and line["trusted"] is True and "minimap" in line
        assert (tmp_path / "rec" / f"{line['i']:05d}_tilt.png").exists()   # the tilt dial's crop too
    finally:
        root.destroy()


def test_shots_csv_gets_the_gun_source_and_keeps_old_rows(tmp_path):
    # 2026-10-12: with the minimap the gun's age is ~0 s; where the gun came from now has a column,
    # added to an older shots.csv without losing a row
    import artillery
    path = tmp_path / "shots.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("time,weapon,gun_x\n2026-09-30 18:51:44,spg,74.86\n")
    artillery.log_shot({"time": "2026-10-12 10:00:00", "weapon": "spg", "gun_x": 65.7,
                        "gun_source": "minimap", "f2_age_s": 42.0}, str(path))
    rows = read_rows(path)
    assert [r["gun_x"] for r in rows] == ["74.86", "65.7"]
    assert rows[1]["gun_source"] == "minimap" and rows[1]["f2_age_s"] == "42.0"
    assert not (tmp_path / "shots.csv.tmp").exists()


def test_an_f4_row_says_where_the_gun_came_from(tmp_path, monkeypatch):
    import math
    app, root = _app(tmp_path, monkeypatch)
    try:
        s = _aimed(app)
        g = app.state["gun"]
        a = math.radians(s.azimuth_deg)
        row = _fire(app, (g[0] + math.sin(a) * s.distance_m / 100, g[1] + math.cos(a) * s.distance_m / 100))
        assert row["gun_source"] == "F2" and float(row["f2_age_s"]) < 5
    finally:
        root.destroy()


def test_recording_saves_every_minimap_read_with_its_result(tmp_path, monkeypatch):
    # 2026-10-12: drives recorded, so confidence over time can be tuned on a real one
    import json
    import hud_bearing
    app, root = _app(tmp_path, monkeypatch)
    try:
        app.mm_recorder = hud_bearing.Recorder(str(tmp_path / "mm"))
        app.locator = _Locator([_fix(65.70, 69.79)])
        app.step()
        assert (tmp_path / "mm" / "00000_minimap.png").exists()
        e = json.loads((tmp_path / "mm" / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert e["map"] == "zestafona" and e["x"] == 65.70 and "why" in e
    finally:
        root.destroy()


def test_a_failed_tape_read_is_recorded_too(tmp_path, monkeypatch):
    import json
    import numpy as np
    import hud_bearing
    app, root = _app(tmp_path, monkeypatch)
    try:
        _aimed(app)
        app.recorder = hud_bearing.Recorder(str(tmp_path / "rec"))
        app.reader.last_tape, app.reader.last_box = np.zeros((90, 720, 3), np.uint8), None
        app.reader.bearing = None                            # no tape: the map is open
        app.step()
        e = json.loads((tmp_path / "rec" / "log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert e["state"] == "notape" and e["why"]
    finally:
        root.destroy()


def test_tracking_runs_on_a_steady_clock_and_the_gun_time_on_the_wall(tmp_path, monkeypatch):
    # 2026-10-12: Windows' time sync can step the wall clock; the tracker and the minimap's
    # freshness use time.monotonic(), while the saved gun time stays wall-clock for shots.csv
    import artillery
    app, root = _app(tmp_path, monkeypatch)
    try:
        _aimed(app)
        app.locator = _Locator([_fix(65.70, 69.79)])
        seen = []
        real = app.tracker.update
        app.tracker.update = lambda now, *a, **k: seen.append(now) or real(now, *a, **k)
        app.reader.bearing = 47.0
        monkeypatch.setattr(artillery.time, "monotonic", lambda: 1000.0)
        app.step()
        assert seen and seen[-1] == 1000.0
        assert app.state["gun_time"] > 1e9                   # the wall clock, not 1000
    finally:
        root.destroy()


def test_an_old_icon_heading_is_not_trusted(tmp_path, monkeypatch):
    # review 2026-10-12: a heading up to 2 s old could put a turning hull's prior past 7.5 deg off
    import time
    import minimap
    monkeypatch.setattr(minimap, "icon_heading", lambda crop: (None, "no tank icon"))
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=10.0))
    try:
        _aimed(app)
        app.icon_heading = (37.0, time.monotonic() - 1.0)     # read a second ago
        seen = []
        real = app.tracker.update
        app.tracker.update = lambda now, frac, ev, prior, trusted=False, proof=None: seen.append(trusted) or \
            real(now, frac, ev, prior, trusted, proof)
        app.reader.bearing = 47.0
        app.step()
        assert seen and seen[-1] is False                     # the target's azimuth, untrusted
    finally:
        root.destroy()


def test_the_state_saves_numpy_numbers_and_a_failed_save_keeps_the_old_file(tmp_path, monkeypatch):
    # 2026-10-12 live: a float32 minimap position failed json.dump halfway, leaving the state file
    # cut off at '"gun": [' -- everything in it lost on the next start
    import numpy as np
    import artillery
    monkeypatch.setattr(artillery, "STATE", str(tmp_path / "state.json"))
    artillery.save_state({"map": "ozeti", "gun": [np.float32(74.93), np.float32(85.42)]})
    st = artillery.load_state()
    assert st["map"] == "ozeti" and abs(st["gun"][0] - 74.93) < 1e-4
    artillery.save_state({"map": "zestafona", "gun": [object()]})
    assert artillery.load_state()["map"] == "ozeti"


def test_a_minimap_fix_stores_plain_numbers(tmp_path, monkeypatch):
    import types
    import numpy as np
    app, root = _app(tmp_path, monkeypatch)
    try:
        fix = types.SimpleNamespace(map="ozeti", x=np.float32(74.93), y=np.float32(85.42), heading=None,
                                    score=np.float32(0.5), lead=np.float32(0.3))
        app.take_minimap(fix, "", 0.0)
        assert all(type(v) is float for v in app.state["gun"])
    finally:
        root.destroy()


def test_a_recorded_bearing_read_saves_the_top_ribbon_too(tmp_path, monkeypatch):
    # USER 2026-10-12: the top ribbon's white labels read where the tape's black ones cannot --
    # recorded so a reader for them can be built and checked
    import numpy as np
    import hud_bearing
    app, root = _app(tmp_path, monkeypatch, tilt=_nose_up(0.0, turret=10.0))
    try:
        _aimed(app)
        app.recorder = hud_bearing.Recorder(str(tmp_path / "rec"))
        app.reader.last_tape, app.reader.last_box = np.zeros((90, 720, 3), np.uint8), None
        app.reader.bearing = 47.0
        app.step()
        assert list((tmp_path / "rec").glob("*_ribbon.png"))
    finally:
        root.destroy()


# --- live 2026-10-12 09:06: the tilt dial's turret read 145 on a quarter of reads (true ~3) --------
def _turret_app(tmp_path, monkeypatch, tilts, icon=None):
    import minimap
    monkeypatch.setattr(minimap, "icon_heading", lambda crop: (icon, "") if icon is not None else (None, "no icon"))
    app, root = _app(tmp_path, monkeypatch, tilt=tilts[0])
    _aimed(app)
    app.reader.tilts = list(tilts)
    app.reader.bearing = 105.0
    return app, root


def test_a_turret_misread_against_the_minimap_heading_is_not_taken(tmp_path, monkeypatch):
    good, bad = _nose_up(1.0, turret=5.0), _nose_up(1.0, turret=145.0)
    app, root = _turret_app(tmp_path, monkeypatch, [good, bad] * 10 + [good], icon=100.0)
    try:
        for _ in range(18):
            app.step()
            assert abs(app.tilt.turret_deg - 5.0) < 1.0, app.tilt.turret_deg
    finally:
        root.destroy()


def test_a_turret_misread_breaking_the_turrets_smooth_turn_is_not_taken(tmp_path, monkeypatch):
    good, bad = _nose_up(1.0, turret=5.0), _nose_up(1.0, turret=145.0)
    app, root = _turret_app(tmp_path, monkeypatch, [good, good, bad, good, bad, bad, good] * 3)
    try:
        for _ in range(18):
            app.step()
            assert abs(app.tilt.turret_deg - 5.0) < 1.0, app.tilt.turret_deg
    finally:
        root.destroy()


def test_a_wrong_first_turret_read_does_not_lock_in(tmp_path, monkeypatch):
    good, bad = _nose_up(1.0, turret=5.0), _nose_up(1.0, turret=145.0)
    app, root = _turret_app(tmp_path, monkeypatch, [bad] + [good] * 12)
    try:
        for _ in range(12):
            app.step()
        assert abs(app.tilt.turret_deg - 5.0) < 1.0
    finally:
        root.destroy()


def _screen_reader(height, monkeypatch):
    """artillery.Reader over a fake 16:9 screen `height` px tall: a horizontal ramp plus a
    vertical one, so any wrong offset or size shows in the pixels a box comes back with."""
    import numpy as np
    import artillery
    w = height * 16 // 9
    ys, xs = np.mgrid[0:height, 0:w]
    screen = np.dstack([(xs * 255 // w), (ys * 255 // height), np.zeros_like(xs), np.full_like(xs, 255)]).astype(np.uint8)

    class Sct:
        grabs = []

        def grab(self, r):
            self.grabs.append(r)
            return screen[r["top"]:r["top"] + r["height"], r["left"]:r["left"] + r["width"]]

    r = artillery.Reader.__new__(artillery.Reader)
    r.sct, r.virtual = Sct(), {"left": 0, "top": 0, "width": w, "height": height}
    r.game = dict(r.virtual)
    r.scale = height / artillery.REF_HEIGHT
    monkeypatch.setattr(artillery, "cursor", lambda: (w // 2, height // 2))
    return r, screen


def test_reader_grabs_unscaled_at_1440p(monkeypatch):
    import hud_mil
    r, screen = _screen_reader(1440, monkeypatch)
    x, y, w, h = hud_mil.BOX
    assert (r.grab_box(hud_mil.BOX) == screen[y:y + h, x:x + w, :3]).all()
    assert r.sct.grabs[-1] == {"left": x, "top": y, "width": w, "height": h}


def test_reader_resamples_other_16x9_screens_to_the_reference_box(monkeypatch):
    import hud_bearing, hud_mil, hud_tilt, minimap
    ref, _ = _screen_reader(1440, monkeypatch)
    for height in (1080, 2160):
        r, _ = _screen_reader(height, monkeypatch)
        for box in (hud_mil.BOX, hud_bearing.BOX, hud_tilt.BOX, minimap.MM):
            got, want = r.grab_box(box).astype(int), ref.grab_box(box).astype(int)
            assert got.shape == want.shape
            assert abs(got - want).max() <= 3, (height, box)     # the ramps: ~1 px of offset at most
        assert r.grab_game().shape == (1440, 2560, 3)


def test_reader_map_read_finds_the_cursor_in_the_resampled_crop(monkeypatch):
    import artillery
    import map_coords
    seen = {}
    monkeypatch.setattr(map_coords, "read", lambda crop, cursor, origin=(0, 0): seen.update(
        shape=crop.shape, cursor=cursor) or map_coords.Reading(None, None))
    dx, dy, w, h = map_coords.GRAB
    for height in (1080, 1440, 2160):
        r, _ = _screen_reader(height, monkeypatch)
        r.read()
        assert seen["shape"] == (h, w, 3)
        assert seen["cursor"] == (-dx, -dy), height
