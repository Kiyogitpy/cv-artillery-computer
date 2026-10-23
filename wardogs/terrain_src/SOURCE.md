# Terrain source chunks (2 m)

- `<map>/manifest.json` + `<map>/chunks/<cx>_<cy>.bin` for `bakurani`, `ozeti`, `zestafona`:
  256 chunks each (16x16), format `wardogs-landscape-collision-u16-v1`, 511x511 vertices 2 m apart per
  chunk, raw little-endian uint16 normalised to each chunk's `minLocalZ`..`maxLocalZ`. 386 MB in all.
  How they decode, and the map-unit mapping, is documented at the top of `terrain.py`.
- Copied on 2026-09-30 from `public/terrain/<map>/` of https://github.com/ifBars/warboard (no licence
  file). Its README says the data comes from https://github.com/apollyon-sys/wardogs-calculator
  (pinned commit `c3252c9d...`), whose CDN terms do not allow bulk download, mirroring or
  redistribution, and that the MIT licence does not cover WARDOGS game assets. Personal use only;
  do not redistribute these files.
- All 768 chunks were checked against the manifests' size and SHA-256 (`terrain.load(map).verify()`,
  also a test). Three Bakurani chunks (`23_19`, `29_19`, `27_22`) came through the download tool
  re-encoded as text and were re-fetched as base64 git blobs; if you re-download, use blobs.

## Checked

- The decode and mapping reproduce the 10 m grids of doubletap-dave/wardogs-arty-buddy
  (an independent decode) to 0.03 m rms, 0.05 m at worst, at 9,000 random points over the three maps.
- The manifests carry `evidence: VERIFIED` and a note that the Y sign was corrected after overlaying
  the terrain on top-view tiles of the map.

## NOT checked

- **Against real impacts.** The ASLs logged in `shots.csv` (23 unique targets) do not correlate with
  these heights under any flip or swap of the axes (best +0.36), 10 m or 2 m alike, so they cannot
  confirm the registration -- nor refute it (the logged ASLs are of unclear provenance, and the gun
  height factor is itself "fitted, not understood"). `shots.csv` now logs `map` and `grid_dz_m` with
  every F8 so real impacts can settle it.
- **Offset datum.** Absolute heights are hundreds of metres from any in-game ASL. Only differences
  (target minus gun) mean anything.
