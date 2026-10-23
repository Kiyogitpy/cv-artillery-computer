"""terrain: the 2 m height chunks. Run: python -m pytest"""
import hashlib
import json

import numpy as np
import pytest

import terrain


def _map(tmp_path, chunk_xs=(0, 1), chunk_ys=(0, 1), lo=0.0, hi=65535.0, scale=1.0, offset=0.0,
         height=lambda ix, iy: 100 * iy + ix, fmt=terrain.FORMAT):
    """A tiny map "m": 2-quad chunks (3x3 vertices). With lo=0, hi=65535, scale=1, offset=0 a stored
    uint16 IS its height in metres, and height(ix, iy) is the height of global vertex (ix, iy).
    Map units: 10 quads per unit in x, and y = (4 - qy) / 10, i.e. y grows as the row index falls."""
    root = tmp_path / "m"
    (root / "chunks").mkdir(parents=True)
    chunks = {}
    for cx in chunk_xs:
        for cy in chunk_ys:
            data = np.array([[height(cx * 2 + c, cy * 2 + r) for c in range(3)] for r in range(3)], dtype="<u2")
            raw = data.tobytes()
            name = f"chunks/{cx}_{cy}.bin"
            (root / name).write_bytes(raw)
            chunks[f"{cx},{cy}"] = {"file": name, "bytes": len(raw), "minLocalZ": lo, "maxLocalZ": hi,
                                    "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {"format": fmt, "mapId": "m", "chunkXMin": min(chunk_xs), "chunkXMax": max(chunk_xs),
                "chunkYMin": min(chunk_ys), "chunkYMax": max(chunk_ys), "chunkQuads": 2, "verticesPerSide": 3,
                "globalQuadOffsetX": 0, "globalQuadOffsetY": 4, "gameUnitsToLandscapeQuadsX": 10,
                "gameUnitsToLandscapeQuadsY": -10, "worldZOffsetMeters": offset,
                "worldZScaleMetersPerLocalUnit": scale, "chunks": chunks}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(tmp_path)


def test_a_vertex_reads_back_in_metres(tmp_path):
    g = terrain.load("m", _map(tmp_path))
    assert g.elevation(0.0, 0.4) == pytest.approx(0)            # qx 0, qy 0
    assert g.elevation(0.4, 0.4) == pytest.approx(4)            # qx 4, qy 0
    assert g.elevation(0.0, 0.0) == pytest.approx(400)          # qx 0, qy 4
    assert g.elevation(0.3, 0.1) == pytest.approx(303)          # qx 3, qy 3


def test_between_vertices_it_interpolates(tmp_path):
    g = terrain.load("m", _map(tmp_path))
    assert g.elevation(0.05, 0.2) == pytest.approx(200.5)       # qx 0.5, qy 2
    assert g.elevation(0.25, 0.15) == pytest.approx(100 * 2.5 + 2.5)


def test_it_is_continuous_across_a_chunk_seam(tmp_path):
    g = terrain.load("m", _map(tmp_path))
    for y in (0.05, 0.2, 0.35):
        qy = 4 - 10 * y
        assert g.elevation(0.2, y) == pytest.approx(100 * qy + 2)              # exactly on the seam
        assert g.elevation(0.2 - 1e-9, y) == pytest.approx(100 * qy + 2, abs=1e-4)
        assert g.elevation(0.2 + 1e-9, y) == pytest.approx(100 * qy + 2, abs=1e-4)


def test_y_grows_north(tmp_path):
    g = terrain.load("m", _map(tmp_path, height=lambda ix, iy: 1000 - 100 * iy))   # rows go up as iy falls
    assert g.elevation(0.2, 0.35) > g.elevation(0.2, 0.05)


def test_the_stored_number_is_normalised_to_the_chunks_own_range(tmp_path):
    root = _map(tmp_path, chunk_xs=(0,), chunk_ys=(0,), lo=-20.0, hi=-10.0, scale=9.0, offset=0.5,
                height=lambda ix, iy: 0 if (ix, iy) == (0, 0) else 65535)
    g = terrain.load("m", root)
    assert g.elevation(0.0, 0.4) == pytest.approx(0.5 + -20.0 * 9.0)     # u16 0 -> minLocalZ
    assert g.elevation(0.2, 0.2) == pytest.approx(0.5 + -10.0 * 9.0)     # u16 65535 -> maxLocalZ


def test_the_exact_edge_counts_and_a_hair_past_it_does_not(tmp_path):
    g = terrain.load("m", _map(tmp_path))
    assert g.elevation(0.4, 0.0) is not None and g.elevation(0.0, 0.4) is not None
    assert g.elevation(-0.01, 0.2) is None and g.elevation(0.41, 0.2) is None
    assert g.elevation(0.2, -0.01) is None and g.elevation(0.2, 0.41) is None


def test_a_missing_chunk_makes_the_spots_it_touches_none(tmp_path):
    root = _map(tmp_path)
    m = json.loads((tmp_path / "m" / "manifest.json").read_text())
    del m["chunks"]["1,1"]
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(m))
    g = terrain.load("m", root)
    assert g.elevation(0.3, 0.1) is None            # inside chunk (1,1): x 0.2-0.4, y 0-0.2
    assert g.elevation(0.1, 0.3) is None            # its vertices at the seam belong to the missing chunk
    assert g.elevation(0.05, 0.35) is not None      # well inside chunk (0,0)


def test_extent_is_in_map_units(tmp_path):
    x0, x1, y0, y1 = terrain.load("m", _map(tmp_path)).extent()
    assert (x0, x1) == pytest.approx((0.0, 0.4)) and (y0, y1) == pytest.approx((0.0, 0.4))


def test_grid_dz_is_target_minus_gun(tmp_path):
    d = _map(tmp_path)
    assert terrain.grid_dz("m", (0.0, 0.4), (0.4, 0.0), d) == pytest.approx(404)
    assert terrain.grid_dz("m", (0.4, 0.0), (0.0, 0.4), d) == pytest.approx(-404)


def test_grid_dz_is_none_when_a_spot_is_outside_or_files_are_missing(tmp_path):
    d = _map(tmp_path)
    assert terrain.grid_dz("m", (0.0, 0.0), (5.0, 5.0), d) is None
    assert terrain.grid_dz("nope", (0.0, 0.0), (0.1, 0.1), d) is None


def test_an_unexpected_format_is_refused(tmp_path):
    root = _map(tmp_path, fmt="something-else")
    with pytest.raises(ValueError):
        terrain.load("m", root)
    assert terrain.grid_dz("m", (0.1, 0.1), (0.2, 0.2), root) is None       # the caller just gets None


def test_a_chunk_file_of_the_wrong_size_is_refused(tmp_path):
    root = _map(tmp_path)
    (tmp_path / "m" / "chunks" / "0_0.bin").write_bytes(b"\x00\x00" * 5)
    with pytest.raises(ValueError):
        terrain.load("m", root).elevation(0.05, 0.35)


def test_verify_finds_a_corrupted_chunk(tmp_path):
    root = _map(tmp_path)
    g = terrain.load("m", root)
    assert g.verify() == []
    p = tmp_path / "m" / "chunks" / "1_0.bin"
    p.write_bytes(p.read_bytes()[:-1] + b"\xff")
    assert g.verify() == ["chunks/1_0.bin"]


# --- the shipped data --------------------------------------------------------------------------

# Heights (m) at 10 m grid nodes, from the 10 m grids of doubletap-dave/wardogs-arty-buddy, which were
# decoded independently of terrain.py; the 2 m data must agree with them to decimetre rounding.
REFERENCE = {
    "bakurani": [((60.0, 60.0), -972.7), ((100.3, 45.7), -857.4), ((20.2, 140.9), -682.7),
                 ((80.0, 80.0), -918.8), ((140.4, 20.6), -630.5)],
    "ozeti": [((60.0, 61.62), -982.7), ((100.3, 47.32), -824.2), ((20.2, 142.52), -560.5),   # that grid's y starts at 1.62
              ((80.0, 81.62), -878.1), ((140.4, 22.22), -745.3)],
    "zestafona": [((60.0, 60.0), 216.0), ((100.3, 45.7), 229.3), ((20.2, 140.9), 327.2),
                  ((80.0, 80.0), 115.7), ((140.4, 20.6), 310.5)],
}


@pytest.mark.parametrize("map_id", terrain.MAPS)
def test_every_shipped_chunk_matches_its_manifest_hash(map_id):
    assert terrain.load(map_id).verify() == []


@pytest.mark.parametrize("map_id", terrain.MAPS)
def test_the_shipped_heights_match_the_independent_10m_reference(map_id):
    g = terrain.load(map_id)
    for (x, y), expected in REFERENCE[map_id]:
        assert g.elevation(x, y) == pytest.approx(expected, abs=0.08), (map_id, x, y)


@pytest.mark.parametrize("map_id", terrain.MAPS)
def test_each_shipped_map_covers_about_163_units_square(map_id):
    x0, x1, y0, y1 = terrain.load(map_id).extent()
    assert 163.0 < x1 - x0 < 163.5 and 163.0 < y1 - y0 < 163.5
    assert terrain.load(map_id).elevation((x0 + x1) / 2, (y0 + y1) / 2) is not None


def test_measured_map_to_grid_offsets():
    # Bakurani: 9 tilt stations, 2026-09-30. Zestafona: 12 F5 captures as tilt stations, hull
    # heading from the minimap icon, 2026-10-12 (all within 2.0 deg at the offset, 2.4-21.5 without).
    assert terrain.MAP_OFFSET_M["bakurani"] == (-28.0, -14.0)
    assert terrain.MAP_OFFSET_M["zestafona"] == (-30.0, -34.0)
    assert "ozeti" not in terrain.MAP_OFFSET_M          # 4 stations: not pinned down yet
