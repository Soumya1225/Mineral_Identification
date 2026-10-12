"""Step 4: validate endmember extraction against GROUND TRUTH (Cuprite benchmark, 12 minerals).

The standard Cuprite unmixing benchmark (250 x 190 pixels, 188 bands) comes with the true spectra of 12 minerals.
VCA is run 10 times; each extracted endmember is paired with one true mineral (Hungarian matching on the spectral
angle) and the spectral angle distance (SAD, radians) is reported per mineral. Lower is better; published VCA
results on this benchmark are around 0.10-0.13 rad mean SAD.

    python -m pipeline.s04_benchmark
"""
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

from pipeline.common import DATASETS, RESULTS, spectral_angles, vca

BENCH = DATASETS / "Cuprite_Benchmark"
RUNS = 10


def _arrays(mat):
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def _band_selection(*mats):
    """The 188 bands kept from the 224 AVIRIS bands (benchmark files store this list; else our own list)."""
    for m in mats:
        for v in m.values():
            if isinstance(v, np.ndarray) and v.dtype.kind in "iuf" and v.size == 188 and v.ndim <= 2:
                idx = np.asarray(v).ravel().astype(int)
                if idx.max() <= 224 and len(np.unique(idx)) == 188:
                    return idx - 1 if idx.min() >= 1 else idx
    from pipeline.common import GOOD
    return GOOD


def load():
    d = _arrays(loadmat(BENCH / "Cuprite_data_R188.mat"))
    g = _arrays(loadmat(BENCH / "Cuprite_GT_nEnd12.mat"))
    print("   data file:", {k: getattr(v, "shape", None) for k, v in d.items()})
    print("   ground-truth file:", {k: getattr(v, "shape", None) for k, v in g.items()})
    Y = next(v for v in d.values() if isinstance(v, np.ndarray) and v.ndim == 2 and v.size > 100000)
    Y = Y.T if Y.shape[0] < Y.shape[1] else Y                 # -> (pixels, bands)
    M = next(v for v in g.values() if isinstance(v, np.ndarray) and v.ndim == 2 and 12 in v.shape
             and max(v.shape) in (188, 224) and v.dtype.kind == "f")
    M = M.T if M.shape[0] != 12 else M                        # -> (12, bands)
    if M.shape[1] != Y.shape[1]:
        sel = _band_selection(d, g)
        if M.shape[1] == 224:
            M = M[:, sel]
        if Y.shape[1] == 224:
            Y = Y[:, sel]
    names = None
    for v in g.values():
        if isinstance(v, np.ndarray) and v.dtype.kind in "OU" and v.size == 12:
            names = [str(np.squeeze(x)).strip() for x in v.ravel()]
    if names is None:
        names = [f"Mineral {i + 1}" for i in range(12)]
    return Y.astype(np.float64), M.astype(np.float64), names


def main():
    Y, M, names = load()
    print(f"Benchmark: {Y.shape[0]:,} pixels x {Y.shape[1]} bands, {len(M)} ground-truth minerals: {names}")
    sads, best = [], None
    for run in range(RUNS):
        E = Y[vca(Y, n=len(M), seed=run)]
        A = spectral_angles(M, E)                             # (12 true, 12 extracted)
        r, c = linear_sum_assignment(A)
        s = A[r, c]
        sads.append(s)
        if best is None or s.mean() < best[0].mean():
            best = (s, E[c])
    sads = np.array(sads)
    table = pd.DataFrame({"mineral": names, "sad_mean_rad": sads.mean(0).round(4), "sad_std": sads.std(0).round(4),
                          "sad_best_run": best[0].round(4)})
    table.to_csv(RESULTS / "benchmark_vca_sad.csv", index=False)
    summary = {"dataset": "Cuprite benchmark (250x190, 188 bands, 12 ground-truth minerals)", "method": "VCA",
               "runs": RUNS, "mean_sad_rad": round(float(sads.mean()), 4), "std_over_runs": round(float(sads.mean(1).std()), 4),
               "best_run_mean_sad": round(float(best[0].mean()), 4)}
    (RESULTS / "benchmark_vca.json").write_text(json.dumps(summary, indent=2))
    print(table.to_string(index=False))
    print(f"\nMean SAD over {RUNS} runs: {summary['mean_sad_rad']} rad (+/- {summary['std_over_runs']}); "
          f"best run {summary['best_run_mean_sad']} rad")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    x = np.arange(M.shape[1])
    for i, ax in enumerate(axes.ravel()):
        ax.plot(x, M[i] / np.linalg.norm(M[i]), "k", lw=1.5, label="ground truth")
        ax.plot(x, best[1][i] / np.linalg.norm(best[1][i]), "r--", lw=1.2, label="VCA")
        ax.set_title(f"{names[i]} (SAD {best[0][i]:.3f})", fontsize=9)
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Cuprite benchmark: VCA endmembers vs ground-truth mineral spectra (band index)")
    fig.tight_layout(); fig.savefig(RESULTS / "06_benchmark_endmembers.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
