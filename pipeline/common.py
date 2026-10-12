"""Shared settings and building blocks: AVIRIS reader, bad-band removal, USGS library resampling, SAM, VCA."""
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "Datasets"                     # every dataset of the project lives here
AVIRIS = DATASETS / "AVIRIS_Cuprite" / "f970619t01p02r02c"
LIBRARY = DATASETS / "USGS_Spectral_Library" / "usgs_splib07a" / "ASCIIdata_splib07a"
PROCESSED = DATASETS / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"            # PNG figures for the report / slides
TABLES = RESULTS / "tables"              # CSV / JSON metrics
MAPS = RESULTS / "maps"                  # full-scene classification maps (.npy)
MODELS = ROOT / "models"
for _d in (FIGURES, TABLES, MAPS):
    _d.mkdir(parents=True, exist_ok=True)

SCENE = "f970619t01p02_r02_sc03.a.rfl"     # the classic Cuprite mining-district scene (614 x 512 pixels)
SAMPLES, BANDS = 614, 224
# water-vapour and noisy bands removed (1-based AVIRIS channels), the standard 224 -> 188 band set for Cuprite
BAD_BANDS = list(range(1, 3)) + list(range(104, 114)) + list(range(148, 168)) + list(range(221, 225))
GOOD = np.array([b - 1 for b in range(1, BANDS + 1) if b not in BAD_BANDS])

# minerals mapped at Cuprite in the USGS Tetracorder / Clark et al. SWIR maps; the classes of this project
MINERALS = ["Alunite", "Buddingtonite", "Calcite", "Chalcedony", "Dickite", "Halloysite", "Jarosite", "Kaolinite",
            "Montmorillonite", "Muscovite"]
SAM_MAX_ANGLE = 0.50                      # radians, between absorption-depth profiles; above -> "unclassified"
# nm: Al-OH (2.16-2.21 um), NH4 (2.12), Fe/Mg-OH (2.26) and carbonate (2.34) absorptions. The 1997 reflectance
# has CO2 artefacts below 2.06 um and a noisy fall-off above 2.40 um, so those parts are left out (tuned on the scene)
SWIR = (2080.0, 2380.0)
MIN_BAND_DEPTH = 0.03                     # weaker absorption than this -> no diagnostic feature, left unclassified


def wavelengths():
    """Band centres and FWHM (nm) of the 188 good bands, from the flight's spectral calibration file."""
    spc = np.loadtxt(next(AVIRIS.glob("*.spc")))
    return spc[GOOD, 0], spc[GOOD, 1]


def load_cube(scene=SCENE):
    """AVIRIS reflectance: 16-bit big-endian integers x 10000, band-interleaved-by-pixel (lines, samples, bands)."""
    raw = np.fromfile(AVIRIS / scene, dtype=">i2")
    lines = raw.size // (SAMPLES * BANDS)
    cube = raw[: lines * SAMPLES * BANDS].reshape(lines, SAMPLES, BANDS)[:, :, GOOD].astype(np.float32) / 10000.0
    return np.clip(cube, 0, 1.5)


# ------------------------------------------------------------------ USGS spectral library
def _read_column(path):
    vals = []
    for line in Path(path).read_text(errors="ignore").splitlines()[1:]:
        try:
            vals.append(float(line.split()[0]))
        except (ValueError, IndexError):
            pass
    return np.array(vals)


def _wavelength_file(spectrum_name):
    tag = re.search(r"_(ASD|BECK|NIC4|AVIRIS)", spectrum_name)
    pat = {"ASD": "*Wavelengths_ASD*", "BECK": "*Wavelengths_BECK*", "NIC4": "*Wavelengths_NIC4*",
           "AVIRIS": "*Wavelengths_AVIRIS*"}.get(tag.group(1) if tag else "", None)
    return next(LIBRARY.glob(pat), None) if pat else None


def resample(wl_um, refl, centres_nm, fwhm_nm):
    """Convolve a library spectrum with Gaussian AVIRIS band responses."""
    wl = wl_um * 1000.0
    ok = (refl > -1) & np.isfinite(refl) & (wl > 0)
    wl, refl = wl[ok], refl[ok]
    out = np.full(len(centres_nm), np.nan)
    for i, (c, f) in enumerate(zip(centres_nm, fwhm_nm)):
        w = np.exp(-0.5 * ((wl - c) / (f / 2.3548)) ** 2)
        if w.sum() > 1e-6:
            out[i] = (w * refl).sum() / w.sum()
    return out


def build_library(centres_nm, fwhm_nm, minerals=MINERALS):
    """Every USGS sample spectrum of each mineral, resampled to the 188 AVIRIS bands. Returns (spectra, labels, names)."""
    folder = next(LIBRARY.glob("ChapterM*"))
    spectra, labels, names = [], [], []
    for k, m in enumerate(minerals):
        files = [p for p in folder.glob(f"splib07a_{m}_*.txt") if re.search(r"_(ASD|BECK)", p.name)]
        for p in files:
            wf = _wavelength_file(p.name)
            if wf is None:
                continue
            s = resample(_read_column(wf), _read_column(p), centres_nm, fwhm_nm)
            if np.isnan(s).mean() < 0.05:
                spectra.append(np.nan_to_num(s, nan=np.nanmean(s)))
                labels.append(k)
                names.append(p.stem)
    return np.array(spectra), np.array(labels), names


# ------------------------------------------------------------------ continuum removal (SWIR)
def swir_index(centres_nm):
    return np.where((centres_nm >= SWIR[0]) & (centres_nm <= SWIR[1]))[0]


def _upper_hull(x, y):
    """Indices of the upper convex hull of one spectrum (the continuum touches these points)."""
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a]) >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def continuum_removed(spectra, centres_nm):
    """SWIR part of each spectrum divided by its convex-hull continuum (Clark & Roush 1984).
    Absorption features become dips below 1.0; brightness and slope differences (illumination, iron) disappear,
    which is why mineral mapping at Cuprite uses this instead of the raw spectrum."""
    idx = swir_index(centres_nm)
    x = centres_nm[idx]
    s = np.asarray(spectra, dtype=np.float64)[..., idx]
    flat = s.reshape(-1, len(idx))
    out = np.ones_like(flat)
    for n, y in enumerate(flat):
        if not np.isfinite(y).all() or y.max() <= 0:
            continue
        h = _upper_hull(x, y)
        out[n] = y / np.maximum(np.interp(x, x[h], y[h]), 1e-6)
    return np.clip(out.reshape(s.shape), 0, 1.5)


# ------------------------------------------------------------------ Spectral Angle Mapper
def spectral_angles(pixels, refs):
    """Angle (radians) between every pixel (N, B) and every reference (R, B)."""
    p = pixels / (np.linalg.norm(pixels, axis=1, keepdims=True) + 1e-12)
    r = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-12)
    return np.arccos(np.clip(p @ r.T, -1, 1))


def sam_classify(cube, refs, ref_labels, n_classes, max_angle=SAM_MAX_ANGLE, depth=None):
    """Per pixel: the mineral whose best library sample has the smallest angle; -1 if above max_angle
    or if the pixel has no absorption deeper than MIN_BAND_DEPTH (depth = per-pixel max band depth)."""
    h, w, b = cube.shape
    px = cube.reshape(-1, b)
    best_angle = np.full(len(px), np.inf)
    best = np.full(len(px), -1)
    for i in range(0, len(px), 20000):
        a = spectral_angles(px[i:i + 20000], refs)
        per_class = np.full((a.shape[0], n_classes), np.inf)
        for k in range(n_classes):
            if (ref_labels == k).any():
                per_class[:, k] = a[:, ref_labels == k].min(axis=1)
        best[i:i + 20000] = per_class.argmin(axis=1)
        best_angle[i:i + 20000] = per_class.min(axis=1)
    best[best_angle > max_angle] = -1
    if depth is not None:
        best[depth.ravel() < MIN_BAND_DEPTH] = -1
    return best.reshape(h, w), best_angle.reshape(h, w)


# ------------------------------------------------------------------ VCA endmember extraction
def vca(pixels, n, seed=0):
    """Vertex Component Analysis (Nascimento & Bioucas-Dias 2005): the n purest pixels of the scene."""
    rng = np.random.default_rng(seed)
    Y = pixels.T                                                   # (B, N)
    mean = Y.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(np.cov(Y), hermitian=True)
    Ud = U[:, :n]
    X = Ud.T @ (Y - mean)                                         # projected, (n, N)
    Xp = np.vstack([X, np.full((1, X.shape[1]), np.abs(X).max())])  # projective form
    idx = np.zeros(n, dtype=int)
    A = np.zeros((n + 1, n + 1))
    A[-1, 0] = 1
    for i in range(n):
        w = rng.normal(size=(n + 1, 1))
        f = w - A @ np.linalg.pinv(A) @ w
        f /= np.linalg.norm(f) + 1e-12
        v = f.T @ Xp
        idx[i] = int(np.argmax(np.abs(v)))
        A[:, i] = Xp[:, idx[i]]
    return idx
