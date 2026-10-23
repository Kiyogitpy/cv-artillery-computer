"""Terrain heights for the three WARDOGS maps, at the source's own 2 m resolution
(wardogs/terrain_src/<map>/, see SOURCE.md there).

Positions are the game's map units (the "x98.99" / "y50.77" the map shows), 100 m
each; +y is north -- the same convention as ballistics.py.

Each map is a 16x16 tile of Landscape collision chunks (wardogs-landscape-collision-u16-v1):
510 quads = 511x511 vertices 2 m apart per chunk, neighbouring chunks sharing their edge
vertices. A chunk file is raw little-endian uint16, row-major, normalised to that chunk's
own range:

    localZ = minLocalZ + u16 / 65535 * (maxLocalZ - minLocalZ)     (both from manifest.json)
    worldZ = worldZOffsetMeters + localZ * worldZScaleMetersPerLocalUnit

and a map position becomes a global vertex position by
    qx = globalQuadOffsetX + gameUnitsToLandscapeQuadsX * x
    qy = globalQuadOffsetY + gameUnitsToLandscapeQuadsY * y      (the Y factor is negative)
with vertex (qx, qy) in chunk (qx // 510, qy // 510), column qx % 510, row qy % 510.
Checked 2026-09-30: this decode and mapping reproduce the 10 m grids of
doubletap-dave/wardogs-arty-buddy to 0.03 m rms at 9,000 random points over the three maps.

The heights sit on an OFFSET datum (hundreds of metres from any in-game ASL), so only
DIFFERENCES between two samples mean anything: grid_dz(map, gun, target) is target height
minus gun height, in metres. The manifests call the mapping VERIFIED (against a top-view
tile overlay); it has NOT been checked against real impacts here -- the ASLs logged in
shots.csv do not correlate with any orientation (best 0.36), so they cannot confirm it.
"""
import hashlib
import json
import os
from collections import OrderedDict

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wardogs", "terrain_src")
MAPS = ("bakurani", "ozeti", "zestafona")
FORMAT = "wardogs-landscape-collision-u16-v1"
EDGE_EPS = 1e-6       # vertices: how far past the mapped edge still counts as on it
CACHE_CHUNKS = 48     # decoded chunks kept (~0.5 MB each)

_grids = {}

# Metres to ADD to a position read off the interactive map to get where the tank really is on
# the grid (east, north). MEASURED 2026-09-30 on Bakurani by grid_align.py: 9 stations (tilt
# dial + compass heading at spots where F5 was pressed on the tank icon) between the base and
# 3 km north agree to 0.5-3.9 deg at this offset and to 1-16 deg without it (chi2 19.4 for 16
# dof); two more were rejected (a road embankment the grid lacks; a steep spot whose heading
# is doubtful). ZESTAFONA, MEASURED 2026-10-12 the same way from 12 artillery F5 captures (F2
# position, tilt dial, hull heading read off the minimap's tank icon -- the minimap is north-up):
# every station within 0.2-2.0 deg at this offset and 2.4-21.5 deg without it, chi2 7.5 for 22 dof,
# the offset within +-2 m. Ozeti has 4 stations so far, not enough to pin it: no shift yet.
MAP_OFFSET_M = {"bakurani": (-28.0, -14.0), "zestafona": (-30.0, -34.0)}


class SourceGrid:
    def __init__(self, directory, manifest):
        if manifest.get("format") != FORMAT:
            raise ValueError(f"unexpected terrain format {manifest.get('format')!r}")
        self.dir, self.man = directory, manifest
        self.q = manifest["chunkQuads"]
        self.n = manifest["verticesPerSide"]
        if self.n != self.q + 1:
            raise ValueError("verticesPerSide must be chunkQuads + 1")
        self._chunks = OrderedDict()      # (cx, cy) -> (uint16 array, minLocalZ, maxLocalZ) or None

    # --- chunks -------------------------------------------------------------------
    def _chunk(self, cx, cy):
        key = (cx, cy)
        if key in self._chunks:
            self._chunks.move_to_end(key)
            return self._chunks[key]
        info = self.man["chunks"].get(f"{cx},{cy}")
        value = None
        if info is not None:
            data = np.fromfile(os.path.join(self.dir, info["file"]), dtype="<u2")
            if data.size != self.n * self.n:
                raise ValueError(f"{info['file']}: {data.size} samples, expected {self.n}x{self.n}")
            value = (data.reshape(self.n, self.n), info["minLocalZ"], info["maxLocalZ"])
        self._chunks[key] = value
        if len(self._chunks) > CACHE_CHUNKS:
            self._chunks.popitem(last=False)
        return value

    def _vertex(self, ix, iy):
        """Height in metres of global vertex (ix, iy), None outside the chunks or in a missing one."""
        m, q = self.man, self.q
        if not (m["chunkXMin"] * q <= ix <= (m["chunkXMax"] + 1) * q
                and m["chunkYMin"] * q <= iy <= (m["chunkYMax"] + 1) * q):
            return None
        cx, cy = min(ix // q, m["chunkXMax"]), min(iy // q, m["chunkYMax"])
        c = self._chunk(cx, cy)
        if c is None:
            return None
        u16, lo, hi = c
        local = lo + int(u16[iy - cy * q, ix - cx * q]) / 65535.0 * (hi - lo)
        return m["worldZOffsetMeters"] + local * m["worldZScaleMetersPerLocalUnit"]

    # --- lookup -------------------------------------------------------------------
    def elevation(self, x, y):
        """Bilinear height in metres at map units (x, y); None outside the covered area or
        where any of the four surrounding vertices has no chunk."""
        m, q = self.man, self.q
        qx = m["globalQuadOffsetX"] + m["gameUnitsToLandscapeQuadsX"] * x
        qy = m["globalQuadOffsetY"] + m["gameUnitsToLandscapeQuadsY"] * y
        x_lo, x_hi = m["chunkXMin"] * q, (m["chunkXMax"] + 1) * q
        y_lo, y_hi = m["chunkYMin"] * q, (m["chunkYMax"] + 1) * q
        if not (x_lo - EDGE_EPS <= qx <= x_hi + EDGE_EPS and y_lo - EDGE_EPS <= qy <= y_hi + EDGE_EPS):
            return None
        qx, qy = min(max(qx, x_lo), x_hi), min(max(qy, y_lo), y_hi)
        ix, iy = min(int(qx), x_hi - 1), min(int(qy), y_hi - 1)   # the cell; the last cell's far edge is x_hi
        fx, fy = qx - ix, qy - iy
        z = (self._vertex(ix, iy), self._vertex(ix + 1, iy), self._vertex(ix, iy + 1), self._vertex(ix + 1, iy + 1))
        if any(v is None for v in z):
            return None
        top = z[0] + (z[1] - z[0]) * fx
        bottom = z[2] + (z[3] - z[2]) * fx
        return top + (bottom - top) * fy

    def extent(self):
        """(x_min, x_max, y_min, y_max) in map units that elevation() covers."""
        m, q = self.man, self.q
        xs = sorted(((m["chunkXMin"] * q - m["globalQuadOffsetX"]) / m["gameUnitsToLandscapeQuadsX"],
                     ((m["chunkXMax"] + 1) * q - m["globalQuadOffsetX"]) / m["gameUnitsToLandscapeQuadsX"]))
        ys = sorted(((m["chunkYMin"] * q - m["globalQuadOffsetY"]) / m["gameUnitsToLandscapeQuadsY"],
                     ((m["chunkYMax"] + 1) * q - m["globalQuadOffsetY"]) / m["gameUnitsToLandscapeQuadsY"]))
        return xs[0], xs[1], ys[0], ys[1]

    def verify(self):
        """Check every chunk file against the manifest's size and SHA-256. Returns the list
        of bad file names (empty when all are intact)."""
        bad = []
        for info in self.man["chunks"].values():
            with open(os.path.join(self.dir, info["file"]), "rb") as f:
                data = f.read()
            if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                bad.append(info["file"])
        return bad


def load(map_id, directory=None):
    """The SourceGrid for a map id ("bakurani", ...), loaded once. `directory` holds one
    folder per map, each with a manifest.json and its chunks/. Raises OSError when the
    files are missing and ValueError/KeyError when they are not what the manifest says."""
    directory = directory or DIR
    key = (directory, map_id)
    if key not in _grids:
        root = os.path.join(directory, map_id)
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        _grids[key] = SourceGrid(root, manifest)
    return _grids[key]


def grid_dz(map_id, gun, target, directory=None):
    """Target height minus gun height in metres, or None when the files are missing or
    either position is outside the covered area. gun and target are (x, y) map units."""
    try:
        grid = load(map_id, directory)
        zg, zt = grid.elevation(*gun), grid.elevation(*target)
    except (OSError, ValueError, KeyError):
        return None
    return None if zg is None or zt is None else zt - zg
