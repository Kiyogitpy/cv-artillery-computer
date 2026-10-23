# Vendored from wardogs-calculator

- Source: https://github.com/apollyon-sys/wardogs-calculator (MIT, see `LICENSE`)
- Commit: `21f2cbc2bd81418d2f6a61678332632fc8bb46e1` (2026-09-28)
- Files: `weapons.json` = `data/weapons.json` and `low-main.json` =
  `data/ballistics/terrain-correction/low-main.json`, byte for byte
  (low-main sha256 97ef8fc98d05d3344c5fb338ac1c9488f9a4766d22d0a385c8735df0b2c7368b).

`ballistics.py` is a line-for-line Python port of the maths that uses this file:

| Port (ballistics.py)   | Original                                                                 |
|------------------------|--------------------------------------------------------------------------|
| `solve()` distance, azimuth | `js/features/results.js` (`Math.hypot(dx, dy)`, `Math.atan2(dx, dy)`, 0-360) |
| `METERS_PER_UNIT = 100`     | `maps/*.json` `coordinateMetersPerUnit` (100 on all three maps)       |
| `in_range`                  | `js/features/weapons.js` `getWeaponElevationSolutions`                |
| `_group`, `interpolate`     | `js/features/weapons.js` `groupBallisticTable`, `interpolateBallisticTable` |
| `format_mil`                | `js/features/results.js` `formatMilSolution`                          |
| `low_arc_height_mil`        | `js/features/experimental-terrain-correction.js` `resolveLowMain` -- read CONTINUOUSLY (interpolated between its 10-mil boundaries, envelope midpoint) instead of snapped to a 10-mil command, and without its +-40 m domain cut: three logged shots at targets ~100 m up landed within ~20 m of where it puts them (one with the target height typed, two assuming it from a neighbouring target) |

Also vendored (2026-09-30): `high-v2.json` = `data/ballistics/terrain-correction/high-v2.json`, byte for byte
(sha256 42eb2bb6a5b2f24aeb04eef4f8fcba65f3aca22ada0f4af9486101f119a39034, identical on `main` 8fa2773).
Upstream status: DISABLED_RESEARCH_CANDIDATE, held-out validation required. `ballistics.high_arc_height_mil`
ports `experimental-terrain-correction.js` `resolveHigh`/`highEnvelope`/`interpolateSparseSegment`, read
continuously like the low arc and applied as a CHANGE from its own level-ground command (so dz 0 is the
flat table). Heights come from the terrain grid (`terrain.py`), not ASL readouts.

Not ported: the low-tail height surface and the experimental SPH-2 hull-pitch correction.

To update: copy the new `data/weapons.json` over this one, then re-read the four
JS functions above for changes before trusting the port.
