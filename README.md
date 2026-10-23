# CV Artillery Computer

A firing computer for the SPH-2 self-propelled gun in **Wardogs**. It uses computer vision
to read the game screen and shows the firing solution in a small window beside the game.

**It only reads the screen.** It never types, clicks or moves the mouse. Aiming and firing stay
with you.

## What it reads

| On screen | Module | Gives |
|---|---|---|
| The big map's cursor readout (`x98.99 y50.77`) | `map_coords.py` | gun, target and impact positions |
| The minimap | `minimap.py` | the gun's position, the map, and the hull's heading |
| The gun sight's bearing tape | `hud_bearing.py`, `hud_compass.py` | where the turret points |
| The gun sight's MIL scale | `hud_mil.py` | the elevation set on the gun |
| The tilt dial | `hud_tilt.py` | hull pitch and roll, and turret angle |

From these it works out distance, azimuth and the mils for each arc (`ballistics.py`). It
corrects for the height difference between gun and target using the terrain grid
(`terrain.py`) and for a tilted hull using the tilt dial. Then it shows how far to traverse
and how far to turn the dial.

## Keys

| Key | Where | Does |
|---|---|---|
| F2 | map open, cursor on your gun | store the gun position (the minimap also places it on its own) |
| F3 (hold) | map open, cursor on the target | show the solution to the cursor |
| F4 | map open, cursor on the crater | mark where the shell landed: shows the miss, corrects the next shot, and logs the shot to `shots.csv` |
| F5 | map closed | save the screen with the stored gun position, for checking minimap matching |
| Space | gun view | noted as a shot, so F4 knows what was fired |

Close the window to quit. The window, gun position and map are remembered in
`artillery_state.json`.

## Setup

Windows, Python 3.13, at 2560×1440. The screen boxes in the `hud_*.py` modules are set for that
resolution.

```
pip install -r requirements.txt
python artillery.py
```

### Data you need to supply

- **Terrain heights** (not included: they aren't mine to redistribute) in `wardogs/terrain_src/<map>/`. See
  [`wardogs/terrain_src/SOURCE.md`](wardogs/terrain_src/SOURCE.md) for the format and where it
  comes from. Without them, the mils assume level ground and the window says so.
- **Map terrain exports** for minimap matching are attached to the
  [latest release](https://github.com/Kiyogitpy/cv-artillery-computer/releases/latest)
  (`bakurani_map.png`, `ozeti_map.png`, `zesty_map.png`, 0.4–1.6 GB each). Put them in `maps/`,
  then build the cache once with `python minimap.py --build`. Without them, store the gun with
  F2 by hand.

## Tests

```
python -m pytest
```

Some tests check the readers against recorded screenshots (`test_shots/`) and the terrain data.
Those files aren't included, so those tests fail in a fresh clone. The rest run on their own.

## Licence

MIT, see [`LICENSE`](LICENSE). The vendored calculator tables in `wardogs/` keep their own MIT
licence.

## Credits

The ballistics are a Python port of
[wardogs-calculator](https://github.com/apollyon-sys/wardogs-calculator) (MIT). Its tables are
vendored in `wardogs/`; see [`wardogs/VENDORED.md`](wardogs/VENDORED.md) and
[`wardogs/LICENSE`](wardogs/LICENSE).

The range tables in `wardogs/` are the calculator's measurements; they are not derived from the game's
files. [wardogs-data-docs](https://github.com/Kiyogitpy/wardogs-data-docs) documents how the game itself
flies a 155 mm shell (gravity, drag, launch speed, from the game's data and code, with its sources and
limits) in its
[projectile flight](https://github.com/Kiyogitpy/wardogs-data-docs/blob/main/pipelines/Wardogs_ProjectileFlight_v1.md)
page, and
[checks that model against logged impacts and these tables](https://github.com/Kiyogitpy/wardogs-data-docs/blob/main/docs/evidence/round21-artillery-range-check.md).
The model comes within about 60 m of logged impacts with no fitted parameters under stated assumptions,
and does not reproduce the tables exactly, so the tables here are still the calculator's.

Wardogs is a game by Bulkhead, published by Team17. This project is not affiliated with either.
