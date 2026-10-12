"""Step 3: presentation figures with ONE colour per mineral across every map (no retraining needed).

The 3D-CNN map is shown only where the pixel has a diagnostic SWIR absorption (the same pixels SAM maps);
featureless pixels stay black, as in the reference.

    python -m pipeline.s03_figures
"""
import json

import numpy as np

from pipeline.common import MINERALS, PROCESSED, RESULTS, SAM_MAX_ANGLE
from pipeline.s02_train_3dcnn import LABEL_ANGLE, spatial_split

COLOURS = {"Alunite": "#1f77b4", "Buddingtonite": "#17becf", "Calcite": "#ff7f0e", "Chalcedony": "#ffbb78",
           "Dickite": "#2ca02c", "Halloysite": "#98df8a", "Jarosite": "#d62728", "Kaolinite": "#ff9896",
           "Montmorillonite": "#9467bd", "Muscovite": "#c5b0d5"}


def show(ax, img, title):
    """img: mineral index into MINERALS, -1 = none."""
    rgb = np.zeros(img.shape + (3,))
    for k, m in enumerate(MINERALS):
        rgb[img == k] = [int(COLOURS[m][i:i + 2], 16) / 255 for i in (1, 3, 5)]
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def legend(fig, present):
    from matplotlib.patches import Patch
    handles = [Patch(color=COLOURS[m], label=m) for m in MINERALS if m in present] + [Patch(color="black", label="No diagnostic absorption")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=8, frameon=False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cube = np.load(PROCESSED / "cuprite_cube_188.npy")
    wl = np.load(PROCESSED / "wavelengths_188.npy")
    sam = np.load(PROCESSED / "sam_labels.npy")
    ang = np.load(PROCESSED / "sam_angles.npy")
    classes = json.loads((RESULTS / "metrics.json").read_text())["classes"]
    cnn_sub = np.load(RESULTS / "cnn_prediction_map.npy")
    to_global = np.array([MINERALS.index(c) for c in classes])
    cnn = np.where(cnn_sub >= 0, to_global[np.clip(cnn_sub, 0, None)], -1)
    cnn[sam < 0] = -1                                   # only where SAM found a diagnostic absorption
    conf = np.where((sam >= 0) & (ang <= LABEL_ANGLE), sam, -1)
    _, test_m = spatial_split(*sam.shape)

    rgb = cube[:, :, [np.abs(wl - x).argmin() for x in (640, 550, 460)]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    ax[0].imshow(np.clip(rgb / np.percentile(rgb, 99), 0, 1)); ax[0].set_title("AVIRIS true colour, Cuprite (Nevada)"); ax[0].axis("off")
    show(ax[1], sam, f"SAM mineral map (USGS library, angle <= {SAM_MAX_ANGLE} rad)")
    legend(fig, {MINERALS[k] for k in np.unique(sam) if k >= 0})
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(RESULTS / "01_sam_mineral_map.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    show(ax[0], conf, "Reference: confident SAM pixels (training labels)")
    show(ax[1], cnn, "3D-CNN prediction")
    for a in ax:
        a.contour(test_m, levels=[0.5], colors="white", linewidths=1.0)
    legend(fig, set(classes) | {MINERALS[k] for k in np.unique(conf) if k >= 0})
    fig.suptitle("White outlines = held-out test blocks (never seen in training)", y=0.98)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96)); fig.savefig(RESULTS / "03_reference_vs_3dcnn_map.png", dpi=140); plt.close(fig)

    agree = ((cnn == sam) & (sam >= 0) & test_m).sum() / max(((sam >= 0) & test_m).sum(), 1)
    print(f"Figures saved. 3D-CNN agrees with SAM on {agree * 100:.1f}% of mapped pixels in the test blocks "
          f"(all SAM pixels, not only the confident ones).")


if __name__ == "__main__":
    main()
