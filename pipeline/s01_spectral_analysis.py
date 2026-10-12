"""Step 1: physics-based analysis of the Cuprite scene.

  1. load AVIRIS reflectance, remove 36 water-vapour / noisy bands (224 -> 188)
  2. build the USGS reference library for 12 Cuprite minerals, resampled to the AVIRIS bands
  3. VCA endmember extraction (purest pixels) and identification of each endmember with SAM against the library
  4. SAM mineral map of the whole scene (the reference map used to train and check the 3D-CNN)
SAM works on CONTINUUM-REMOVED SWIR spectra (2000-2450 nm), where these minerals have their diagnostic
absorptions; on raw full-range spectra SAM mostly sees brightness and iron slopes and mislabels the scene.

    python -m pipeline.s01_spectral_analysis
"""
import json

import numpy as np
import pandas as pd

from pipeline.common import (MINERALS, PROCESSED, RESULTS, SAM_MAX_ANGLE, SWIR, build_library, continuum_removed,
                             load_cube, sam_classify, spectral_angles, vca, wavelengths)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("[1] Loading AVIRIS Cuprite scene...")
    cube = load_cube()
    wl, fwhm = wavelengths()
    h, w, b = cube.shape
    print(f"   {h} x {w} pixels, {b} bands ({wl[0]:.0f}-{wl[-1]:.0f} nm), mean reflectance {cube.mean():.3f}")
    np.save(PROCESSED / "cuprite_cube_188.npy", cube)
    np.save(PROCESSED / "wavelengths_188.npy", wl)

    print("[2] USGS spectral library...")
    lib, lib_lab, lib_names = build_library(wl, fwhm)
    per = pd.Series(lib_lab).map(lambda k: MINERALS[k]).value_counts()
    print(f"   {len(lib)} reference spectra:", per.to_dict())
    missing = [m for m in MINERALS if m not in per.index]
    if missing:
        print("   no library spectra found for:", missing)
    np.save(PROCESSED / "library_spectra_188.npy", lib)
    np.save(PROCESSED / "library_labels.npy", lib_lab)

    print("[3] VCA endmember extraction...")
    px = cube.reshape(-1, b)
    valid = np.where(px.mean(axis=1) > 0.02)[0]
    idx = valid[vca(px[valid], n=len(MINERALS))]
    ends = px[idx]
    lib_cr = 1 - continuum_removed(lib, wl)                 # absorption-depth profiles (0 = no absorption)
    ang = spectral_angles(1 - continuum_removed(ends, wl), lib_cr)
    rows = []
    for i, e in enumerate(idx):
        j = int(ang[i].argmin())
        rows.append({"endmember": i + 1, "row": int(e // w), "col": int(e % w),
                     "best_match": MINERALS[lib_lab[j]], "library_sample": lib_names[j],
                     "angle_rad": round(float(ang[i, j]), 4),
                     "confident": bool(ang[i, j] <= SAM_MAX_ANGLE)})
    em = pd.DataFrame(rows)
    em.to_csv(RESULTS / "endmembers_vca.csv", index=False)
    np.save(PROCESSED / "endmembers_vca_188.npy", ends)
    print(em[["endmember", "best_match", "angle_rad", "confident"]].to_string(index=False))

    print("[4] SAM mineral map...")
    cr = continuum_removed(cube, wl).astype(np.float32)
    depth = 1 - cr.min(axis=-1)
    print(f"   continuum-removed SWIR {SWIR[0]:.0f}-{SWIR[1]:.0f} nm: {cr.shape[-1]} bands")
    labels, angles = sam_classify(1 - cr, lib_cr, lib_lab, len(MINERALS), depth=depth)
    ok = depth.ravel() >= 0.02
    print("   best-match angle percentiles (pixels with an absorption):",
          {q: round(float(np.percentile(angles.ravel()[ok], q)), 3) for q in (10, 25, 50, 75, 90)} if ok.any() else "none")
    np.save(PROCESSED / "sam_labels.npy", labels)
    np.save(PROCESSED / "sam_angles.npy", angles)
    dist = pd.Series(labels.ravel()).map(lambda k: MINERALS[k] if k >= 0 else "Unclassified").value_counts()
    dist.rename("pixels").to_csv(RESULTS / "sam_mineral_distribution.csv")
    print(dist.to_string())
    lab_names = [MINERALS[k] for k in range(len(MINERALS))]
    (RESULTS / "spectral_summary.json").write_text(json.dumps({"method": "SAM on continuum-removed SWIR", "classes": lab_names,
        "scene_pixels": [h, w], "bands_used": int(b), "library_spectra": int(len(lib)),
        "sam_max_angle_rad": SAM_MAX_ANGLE, "classified_fraction": round(float((labels >= 0).mean()), 3),
        "endmembers_identified": em["best_match"].tolist()}, indent=2))
    figures(cube, wl, labels, lib, lib_lab, ends, em)


def figures(cube, wl, labels, lib, lib_lab, ends, em):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#000000"] + list(plt.get_cmap("tab20").colors[: len(MINERALS)]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    rgb = cube[:, :, [np.abs(wl - x).argmin() for x in (640, 550, 460)]]
    ax[0].imshow(np.clip(rgb / np.percentile(rgb, 99), 0, 1)); ax[0].set_title("AVIRIS true colour (Cuprite)")
    im = ax[1].imshow(labels + 1, cmap=cmap, vmin=0, vmax=len(MINERALS), interpolation="nearest")
    ax[1].set_title("SAM mineral map (USGS library)")
    cb = fig.colorbar(im, ax=ax[1], ticks=np.arange(len(MINERALS) + 1), fraction=0.046)
    cb.ax.set_yticklabels(["Unclassified"] + MINERALS, fontsize=7)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(RESULTS / "01_sam_mineral_map.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, e in enumerate(ends):
        ax.plot(wl, e + i * 0.15, lw=1)
        ax.text(wl[-1] + 10, e[-1] + i * 0.15, f"E{i + 1}: {em.best_match[i]}", fontsize=7)
    ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance (offset)", title="VCA endmembers and their USGS match")
    fig.tight_layout(); fig.savefig(RESULTS / "02_endmember_signatures.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
