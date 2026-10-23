"""ballistics.py, the port of wardogs-calculator's maths. Run: python -m pytest"""
import json
import os

import cv2
import pytest

import ballistics as b
import map_coords

W = b.load_weapons()
RAW = {w["id"]: w for w in json.load(open(b.WEAPONS_JSON, encoding="utf-8"))["weapons"]}


def test_both_weapons_load_with_their_arcs():
    assert set(W) == {"mortar", "spg"}
    assert set(W["spg"].tables) == {"low", "high"} and set(W["mortar"].tables) == {"single"}
    assert (W["spg"].min_m, W["spg"].max_m) == (780, 2629)
    assert (W["mortar"].min_m, W["mortar"].max_m) == (132, 684)


@pytest.mark.parametrize("target,az", [((0, 10), 0), ((10, 0), 90), ((0, -10), 180), ((-10, 0), 270),
                                       ((10, 10), 45)])
def test_azimuth_is_clockwise_from_north(target, az):
    assert b.solve(W["spg"], (0, 0), target).azimuth_deg == pytest.approx(az)


def test_one_map_unit_is_100_m():
    assert b.solve(W["spg"], (10, 10), (13, 14)).distance_m == pytest.approx(500)


@pytest.mark.parametrize("arc", ["low", "high"])
def test_exact_table_range_gives_its_mil(arc):
    groups = b._group(W["spg"].tables[arc])
    d, mils = next((d, ms) for d, ms in groups if len(ms) == 1 and 800 < d < 2600)
    m = b.interpolate(W["spg"].tables[arc], d)
    assert m.mil == mils[0] and b.format_mil(m) == str(int(mils[0]))


def test_between_two_rows_it_interpolates_linearly():
    groups = [g for g in b._group(W["mortar"].tables["single"]) if len(g[1]) == 1]
    (d0, (m0,)), (d1, (m1,)) = groups[10], groups[11]
    m = b.interpolate(W["mortar"].tables["single"], (d0 + d1) / 2)
    assert m.mil == pytest.approx((m0 + m1) / 2)


def test_several_mils_at_one_range_show_as_a_span_like_the_site():
    groups = b._group(W["spg"].tables["high"])
    d, mils = next((d, ms) for d, ms in groups if len(ms) > 1)
    m = b.interpolate(W["spg"].tables["high"], d)
    assert m.mil is None and b.format_mil(m) == f"{int(min(mils))}–{int(max(mils))}"


def test_out_of_range_has_no_mils():
    s = b.solve(W["mortar"], (0, 0), (0, 8))          # 800 m, the mortar stops at 684
    assert not s.in_range and s.mils == {}
    s = b.solve(W["spg"], (0, 0), (0, 5))             # 500 m, the SPH-2 starts at 780
    assert not s.in_range


def test_sph_low_arc_is_missing_below_its_table_and_high_arc_is_there():
    s = b.solve(W["spg"], (0, 0), (0, 10))            # 1000 m: low table starts at 1181
    assert s.in_range and s.mils["low"] is None and s.mils["high"] is not None


def test_rounding_is_javascripts():
    assert b.js_round(844.5) == 845 and b.js_round(845.5) == 846 and b.js_round(844.49) == 844


def test_the_table_file_is_the_vendored_one_untouched():
    # weapons.json must stay byte-identical to upstream; this pins what the port reads.
    assert RAW["spg"]["ballistics"]["low"][0] == [1181, 20]
    assert RAW["spg"]["ballistics"]["high"][-1] == [735, 1400]


def test_gun_and_target_read_off_two_captures_give_a_solution():
    shots = {os.path.basename(p): (p, at) for p, _, at in map_coords.samples()}
    gun = map_coords.read(cv2.imread(shots["x98.99_y50.77_at979-978.jpg"][0]),
                          shots["x98.99_y50.77_at979-978.jpg"][1])
    tgt = map_coords.read(cv2.imread(shots["x102.37_y59.66_at1363-944.jpg"][0]),
                          shots["x102.37_y59.66_at1363-944.jpg"][1])
    s = b.solve(W["spg"], (gun.x, gun.y), (tgt.x, tgt.y))
    assert s.distance_m == pytest.approx(((102.37 - 98.99) ** 2 + (59.66 - 50.77) ** 2) ** 0.5 * 100)
    assert 0 < s.azimuth_deg < 90 and s.in_range and s.mils["high"] is not None


@pytest.mark.parametrize("impact,want", [
    ((0, 12), (200, 0)),      # firing north: further north is long
    ((0, 8), (-200, 0)),      # short
    ((1, 10), (0, 100)),      # east of a northward line is right
    ((-1, 10), (0, -100)),    # west is left
])
def test_miss_long_short_left_right_firing_north(impact, want):
    got = b.miss((0, 0), (0, 10), impact)
    assert got == pytest.approx(want)


def test_miss_follows_the_line_of_fire_not_the_map_axes():
    # Firing east: north of the line is LEFT, further east is long.
    assert b.miss((0, 0), (10, 0), (10, 1)) == pytest.approx((0, -100))
    assert b.miss((0, 0), (10, 0), (11, 0)) == pytest.approx((100, 0))
    # Firing south-west, 1 unit long.
    s = 2 ** -0.5
    assert b.miss((0, 0), (-10, -10), (-10 - s, -10 - s)) == pytest.approx((100, 0))


def test_miss_with_gun_on_target_is_undefined():
    assert b.miss((5, 5), (5, 5), (6, 6)) is None


def test_height_model_at_level_ground_is_the_flat_table():
    for d in (1300, 1600, 1879.4, 2096.3, 2400):
        flat = b.interpolate(W["spg"].tables["low"], d).mil
        assert b.low_arc_height_mil(d, 0)[0] == pytest.approx(flat, abs=0.5)


def test_height_model_matches_the_users_own_numbers():
    # The user's "2094 m is 280 mil, 2125 is 290" on the low arc, for a target ~80 m up.
    assert b.low_arc_height_mil(2094, 80)[0] == pytest.approx(278, abs=1)
    assert b.low_arc_height_mil(2125, 80)[0] == pytest.approx(288, abs=1)


def _landing(mil, dz):
    """Where the model puts a low-arc shot at `mil` on ground dz above the gun."""
    return min(range(1283, 2440), key=lambda d: abs(b.low_arc_height_mil(d, dz)[0] - mil)
               if b.low_arc_height_mil(d, dz)[0] is not None else 1e9)


@pytest.mark.parametrize("fired,dz,landed", [
    (235.6, 100, 1892.8),   # shots.csv 15:49 -- target next to the ASL-100 one
    (235.6, 100, 1900.8),   # 15:50
    (172.5, 100, 1597.0),   # 15:55, target ASL 100 typed
])
def test_height_model_explains_the_logged_shots(fired, dz, landed):
    assert abs(_landing(fired, dz) - landed) <= 25


def test_height_model_refuses_outside_its_data():
    mil, why = b.low_arc_height_mil(1000, 50)
    assert mil is None and "1283-2439" in why
    mil, why = b.low_arc_height_mil(2400, 400)
    assert mil is None and "beyond" in why


def test_solve_adds_the_low_arc_height_mil_only_when_the_height_is_known():
    s = b.solve(W["spg"], (60.94, 84.57), (67.97, 102.0), dz_m=100)
    assert s.height["low"][0] == pytest.approx(229.5, abs=0.5) and s.mils["low"].mil == pytest.approx(172.5, abs=0.1)
    assert b.solve(W["spg"], (60.94, 84.57), (67.97, 102.0)).height == {}
    assert b.solve(W["mortar"], (0, 0), (0, 5), dz_m=30).height == {}


@pytest.mark.parametrize("arc,long_m,sign", [
    ("low", -30, +1),    # low arc short: raise the barrel
    ("low", +30, -1),
    ("high", -30, -1),   # high arc short: LOWER the barrel (flatter goes further)
    ("high", +30, +1),
])
def test_fall_of_shot_adjustment_direction(arc, long_m, sign):
    d = b.adjust(W["spg"], arc, 2097.6, 0, long_m)
    assert d is not None and d * sign > 0


def test_fall_of_shot_adjustment_size_is_the_table_slope():
    # Low arc near 2100 m: 230 -> 2079 m, 240 -> 2110 m, ~3.1 m/mil; 31 m short ~ +10 mil.
    assert b.adjust(W["spg"], "low", 2094, 0, -31) == pytest.approx(10, abs=0.8)


def test_high_arc_adjustment_ignores_the_height_model_slope():
    # Live 2026-10-13, 1441 m, dz +79.8: the shells moved 0.273 mil/m; high-v2's slope
    # next to its +-80 m clip is 0.72, which overshot every F4 2.7x.
    flat = b.adjust(W["spg"], "high", 1441.3, None, -1.0)
    assert b.adjust(W["spg"], "high", 1441.3, 79.8, -1.0) == pytest.approx(flat)
    assert -flat == pytest.approx(0.273, abs=0.01)


def test_fall_of_shot_adjustment_is_none_where_the_arc_has_no_table():
    assert b.adjust(W["spg"], "low", 900, None, -20) is None


# --- vehicle tilt (hud_tilt's dial) ------------------------------------------------
ONE_DEG = 17.453          # mil (milliradians) in one degree


@pytest.mark.parametrize("pitch,roll,turret,want", [
    (0, 0, 0, 0.0),
    (0, 0, 137, 0.0),
    (1, 0, 0, -ONE_DEG),        # nose up, gun forward: the barrel points 1 deg higher -- dial 1 deg less
    (1, 0, 180, +ONE_DEG),      # nose up, gun over the back: it points 1 deg lower
    (0, 1, 90, -ONE_DEG),       # right side up, gun to the right: higher
    (0, 1, 270, +ONE_DEG),      # right side up, gun to the left: lower
])
def test_tilt_moves_the_dial_by_the_tilt_along_the_gun(pitch, roll, turret, want):
    assert b.tilt_mil(100, pitch, roll, turret) == pytest.approx(want, abs=0.01)


def test_tilt_at_low_elevation_is_about_the_tilt_along_the_gun():
    # 3 deg nose up, 2 deg right side up, gun 30 deg right: (3 cos 30 + 2 sin 30) deg = 62.8 mil
    assert b.tilt_mil(200, 3, 2, 30) == pytest.approx(-62.8, abs=3)


def test_tilt_across_a_steep_barrel_still_costs_elevation():
    # High arc, 1300 mil (74.5 deg), gun square to a 5 deg nose-up hull. The tilt is
    # across the gun, yet the barrel ends up LOWER: asin(sin 1.3 / cos 5deg) - 1.3 rad.
    assert b.tilt_mil(1300, 5, 0, 90) == pytest.approx(14.12, abs=0.05)


def test_an_elevation_the_tilted_gun_cannot_reach_is_none():
    # 1500 mil is 85.9 deg; rolled 10 deg with the gun forward it tops out at 80 deg.
    assert b.tilt_mil(1500, 0, 10, 0) is None


def test_high_arc_height_model_is_the_flat_mil_at_level_ground_and_falls_with_height():
    w = b.load_weapons()["spg"]
    for d in (900, 1123.5, 1537.7, 2000, 2400):
        flat = b.interpolate(w.tables["high"], d).mil
        assert b.high_arc_height_mil(d, 0.0, flat)[0] == flat
        up, down = (b.high_arc_height_mil(d, dz, flat)[0] for dz in (40, -40))
        assert up < flat < down          # a higher target wants a smaller mil (a longer shot)


def test_high_arc_height_model_refuses_outside_its_domain():
    assert b.high_arc_height_mil(500, 0, 1400)[0] is None       # under 780 m
    assert b.high_arc_height_mil(1500, 100, 1200)[0] is None    # beyond +-80 m


def test_solve_gives_the_high_arc_a_height_mil():
    w = b.load_weapons()["spg"]
    s = b.solve(w, (85.95, 54.73), (102.35, 62.87), 30.0)
    assert s.height["high"][0] is not None and s.height["high"][0] < s.mils["high"].mil


def test_a_level_hull_points_the_barrel_where_it_is_set():
    az, el = b.barrel(40.0, 0.0, 0.0, 25.0, 300.0)
    assert az == pytest.approx(65.0, abs=1e-6) and el == pytest.approx(300.0, abs=1e-6)


def test_a_rolled_hull_swings_the_barrel_off_the_turret_heading():
    # the 2026-09-30 shots: roll +12.6 (right side up), barrel at 191 mil, turret 8.5 deg left
    az, el = b.barrel(16.2, -0.3, 12.6, -8.5, 191.16)
    assert az < 7.7 - 1.5                                   # well left of the tape's 7.7
    assert el == pytest.approx(149.2, abs=0.5)              # and lower than the dial says
    # the swing grows with elevation and flips with the side of the roll
    assert b.barrel(0, 0, 12.6, 0, 400)[0] > 180 or b.barrel(0, 0, 12.6, 0, 400)[0] < 356
    assert b.barrel(0, 0, -12.6, 0, 300)[0] < 180 and b.barrel(0, 0, 12.6, 0, 300)[0] > 180


def test_aim_round_trips_through_barrel():
    for heading, p, r, az, el in ((14.2, -0.3, 12.7, 13.02, 364.2), (200.0, 5.0, -8.0, 195.0, 1100.0),
                                  (90.0, 0.0, 0.0, 120.0, 80.0)):
        turret, dial = b.aim(heading, p, r, az, el)
        got_az, got_el = b.barrel(heading, p, r, turret, dial)
        assert ((got_az - az + 180) % 360 - 180) == pytest.approx(0, abs=1e-4)
        assert got_el == pytest.approx(el, abs=1e-3)


def test_aim_on_a_level_hull_is_just_the_azimuth_and_the_elevation():
    turret, dial = b.aim(10.0, 0.0, 0.0, 70.0, 250.0)
    assert turret == pytest.approx(60.0, abs=1e-4) and dial == pytest.approx(250.0, abs=1e-3)
