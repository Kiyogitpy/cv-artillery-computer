"""Calibrate the tilt dial from a slow full turn of the hull on a slope.

    python tilt_fit.py --start-bearing 14            # sample until Ctrl+C, then fit
    python tilt_fit.py --csv debug\\tilt_x.csv        # refit a saved run

Park on the slope in the gun seat (map closed), give the bearing tape's reading as
--start-bearing (within 7 degrees: it is how the 15-degree-ambiguous tape is read), then turn
the HULL through a full circle at any ordinary keyboard turn rate (the tape is followed
sample to sample at 20 Hz, so anything under ~100 deg/s is fine; pauses and extra laps are
fine too) and press Ctrl+C. The hull's heading is the sight bearing minus the turret angle on the dial, so the
turret may point anywhere.

On a plane, with heading h, uphill heading h0 and grade g:
    pitch = p0 + g_p * cos(h - h0)       (ticks)         roll = r0 + g_r * sin(h - h0)   (dial degrees)
so the fit gives the UPHILL HEADING (from the pitch), the ROLL SIGN (+ when the right end
of the bars is higher, and the right side is the higher one on the heading h0 + 90 when
that holds), and the scale between the two: `deg_per_tick = g_r / g_p` if the bars are true
degrees. It does NOT say whether they are: that needs the slope's real angle from somewhere
else. The phase check (roll must lag pitch by 90) and the residuals say whether the dial
follows a plane at all.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COVER_MIN_DEG = 300     # the samples' headings must span this much of the circle
RATE_HZ = 20            # the tape is followed sample to sample, so this bounds the turn: ~7 deg / 0.05 s


def unwrap_bearing(prev, deg):
    """deg (0-360) as the value nearest `prev`, to follow a turn past north."""
    return deg + 360 * round((prev - deg) / 360)


def fit(rows):
    """rows: [(heading_deg, pitch_ticks, roll_deg)]. Returns a dict, or raises ValueError."""
    a = np.array(rows, float)
    h = np.radians(a[:, 0])
    cover = len({int(x) // 30 for x in a[:, 0] % 360}) * 30
    if len(a) < 30 or cover < COVER_MIN_DEG:
        raise ValueError(f"need a (nearly) full turn: {len(a)} samples covering {cover} of 360 degrees")
    X = np.column_stack([np.ones_like(h), np.cos(h), np.sin(h)])
    (p0, pc, ps), *_ = np.linalg.lstsq(X, a[:, 1], rcond=None)
    (r0, rc, rs), *_ = np.linalg.lstsq(X, a[:, 2], rcond=None)
    ap, ar = math.hypot(pc, ps), math.hypot(rc, rs)
    up_h = math.degrees(math.atan2(ps, pc)) % 360          # heading where the pitch peaks
    roll_peak = math.degrees(math.atan2(rs, rc)) % 360     # heading where the roll peaks
    # on a plane the roll is g_r*sin(h-h0): its peak sits 90 deg CLOCKWISE of the uphill heading
    lag = (roll_peak - up_h + 180) % 360 - 180
    res_p = a[:, 1] - X @ [p0, pc, ps]
    res_r = a[:, 2] - X @ [r0, rc, rs]
    return {"n": len(a), "cover_deg": cover, "pitch_zero_ticks": p0, "roll_zero_deg": r0,
            "pitch_amp_ticks": ap, "roll_amp_deg": ar, "uphill_heading": up_h,
            "roll_peak_heading": roll_peak, "roll_phase_vs_pitch_deg": lag,
            "deg_per_tick_if_roll_true": ar / ap if ap > 1e-6 else None,
            "pitch_rms_ticks": float(np.sqrt((res_p ** 2).mean())),
            "roll_rms_deg": float(np.sqrt((res_r ** 2).mean()))}


def report(f):
    lines = [f"{f['n']} samples over {f['cover_deg']} deg of heading",
             f"pitch: zero {f['pitch_zero_ticks']:+.3f} ticks, amplitude {f['pitch_amp_ticks']:.3f} ticks "
             f"(rms {f['pitch_rms_ticks']:.3f})",
             f"roll:  zero {f['roll_zero_deg']:+.2f} deg, amplitude {f['roll_amp_deg']:.2f} deg "
             f"(rms {f['roll_rms_deg']:.2f})",
             f"UPHILL HEADING {f['uphill_heading']:.1f}   (roll peaks at {f['roll_peak_heading']:.1f}; "
             f"phase vs pitch {f['roll_phase_vs_pitch_deg']:+.1f} deg -- +90 = right end up on the right, "
             f"-90 = the bars lean the other way)"]
    if f["deg_per_tick_if_roll_true"] is not None:
        lines.append(f"deg per tick IF the bars are true degrees: {f['deg_per_tick_if_roll_true']:.2f}   "
                     f"(the slope would be {f['roll_amp_deg']:.2f} deg)")
        lines.append("with the slope's real angle g from elsewhere: deg per tick = g / pitch amplitude = "
                     f"g / {f['pitch_amp_ticks']:.3f};  roll scale = g / {f['roll_amp_deg']:.2f}")
    return "\n".join(lines)


def sample(start_bearing, seconds):
    import artillery
    reader = artillery.Reader()
    rows, prev, t_end = [], float(start_bearing), time.time() + seconds
    print(f"sampling {seconds:.0f} s from bearing {start_bearing}: turn the hull slowly through a full "
          f"circle; Ctrl+C to stop early", flush=True)
    try:
        while time.time() < t_end:
            t0 = time.time()
            b, why_b = reader.read_bearing(prev % 360)
            t, why_t = reader.read_tilt()
            if b is not None and t is not None:
                sight = unwrap_bearing(prev, b.deg)
                prev = sight
                rows.append(((sight - t.turret_deg) % 360, t.pitch_ticks, t.roll_deg, sight % 360, t.turret_deg))
            time.sleep(max(0.0, 1 / RATE_HZ - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start-bearing", type=float, help="the bearing tape now (within 7 deg)")
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--csv", help="refit this saved run instead of sampling")
    args = ap.parse_args(argv)
    if args.csv:
        with open(args.csv, newline="") as f:
            rows = [(float(r["heading"]), float(r["pitch_ticks"]), float(r["roll_deg"]))
                    for r in csv.DictReader(f)]
    else:
        if args.start_bearing is None:
            ap.error("--start-bearing is required when sampling")
        from winkeys import dpi_aware
        dpi_aware()
        got = sample(args.start_bearing, args.seconds)
        os.makedirs(os.path.join(HERE, "debug"), exist_ok=True)
        path = os.path.join(HERE, "debug", time.strftime("tilt_%Y%m%d-%H%M%S.csv"))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["heading", "pitch_ticks", "roll_deg", "sight", "turret"])
            w.writerows(got)
        print(f"saved {len(got)} samples -> {path}")
        rows = [r[:3] for r in got]
    try:
        print(report(fit(rows)))
    except ValueError as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
