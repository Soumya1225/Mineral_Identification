"""Step 2: 3D-CNN mineral classifier with an honest SPATIAL train/test split.

Labels: the confident pixels of the SAM / USGS-library reference map (angle <= LABEL_ANGLE).
Split:  the image is cut into 4 x 4 blocks; 4 blocks are held out for testing, and a 4-pixel buffer around them
        is removed from training, so no test pixel (or its 9 x 9 neighbourhood) is ever seen in training.
Models: 3D-CNN (spectral-spatial, on 30 PCA components, 9 x 9 patches) vs Random Forest on the 188-band spectrum.
Also shown: what a RANDOM pixel split would report (the inflated number many papers give).

    python -m pipeline.s02_train_3dcnn
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from pipeline.common import MINERALS, MODELS, PROCESSED, RESULTS

LABEL_ANGLE = 0.40
PATCH, PCS = 9, 30
GRID, BUFFER = 4, PATCH // 2
TEST_BLOCKS = [(0, 3), (1, 1), (2, 2), (3, 0)]       # one block per row and column -> test spread over the scene
MAX_PER_CLASS, MIN_PER_CLASS = 3000, 60
EPOCHS, SEED = 15, 42


def spatial_split(h, w):
    test = np.zeros((h, w), bool)
    bh, bw = h // GRID, w // GRID
    for r, c in TEST_BLOCKS:
        test[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw] = True
    from scipy.ndimage import binary_dilation
    near_test = binary_dilation(test, iterations=BUFFER)
    return ~near_test, test                                  # train mask, test mask


def patches(pcs_img, rows, cols):
    r = BUFFER
    padded = np.pad(pcs_img, ((r, r), (r, r), (0, 0)), mode="reflect")
    out = np.empty((len(rows), PCS, PATCH, PATCH), np.float32)
    for i, (y, x) in enumerate(zip(rows, cols)):
        out[i] = padded[y:y + PATCH, x:x + PATCH].transpose(2, 0, 1)
    return out[:, None]                                      # (N, 1, PCS, 9, 9)


def build_net(n_classes):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv3d(1, 8, (7, 3, 3)), nn.BatchNorm3d(8), nn.ReLU(),
        nn.Conv3d(8, 16, (5, 3, 3)), nn.BatchNorm3d(16), nn.ReLU(),
        nn.Conv3d(16, 32, (3, 3, 3)), nn.BatchNorm3d(32), nn.ReLU(),
        nn.Flatten(), nn.Linear(32 * (PCS - 12) * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, n_classes))


def train_cnn(Xtr, ytr, n_classes):
    import torch
    torch.manual_seed(SEED)
    net = build_net(n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    hist = []
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(X))
        tot, t0 = 0.0, time.time()
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(net(X[b]), y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        hist.append(tot / len(X))
        print(f"   epoch {ep + 1:2d}/{EPOCHS}  loss {hist[-1]:.4f}  ({time.time() - t0:.0f}s)")
    return net, hist


def predict(net, X, bs=2048):
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(net(torch.from_numpy(X[i:i + bs])).argmax(1).numpy())
    return np.concatenate(out)


def predict_at(net, pcs_img, rows, cols, chunk=20000):
    """Predict in chunks so the 9 x 9 x 30 patches never need gigabytes of memory."""
    return np.concatenate([predict(net, patches(pcs_img, rows[i:i + chunk], cols[i:i + chunk]))
                           for i in range(0, len(rows), chunk)]) if len(rows) else np.array([], int)


def scores(name, y, p, classes):
    return {"model": name, "overall_accuracy": round(accuracy_score(y, p), 4), "kappa": round(cohen_kappa_score(y, p), 4),
            "macro_f1": round(f1_score(y, p, average="macro", labels=range(len(classes)), zero_division=0), 4)}


def main():
    rng = np.random.default_rng(SEED)
    cube = np.load(PROCESSED / "cuprite_cube_188.npy")
    sam = np.load(PROCESSED / "sam_labels.npy")
    ang = np.load(PROCESSED / "sam_angles.npy")
    h, w, b = cube.shape
    label = np.where((sam >= 0) & (ang <= LABEL_ANGLE), sam, -1)
    counts = pd.Series(label[label >= 0]).value_counts()
    keep = [k for k in sorted(counts.index) if counts[k] >= MIN_PER_CLASS]
    classes = [MINERALS[k] for k in keep]
    remap = {k: i for i, k in enumerate(keep)}
    y_img = np.vectorize(lambda v: remap.get(v, -1))(label)
    print(f"[1] {len(classes)} classes with >= {MIN_PER_CLASS} confident pixels: {classes}")
    if len(classes) < 2:
        raise SystemExit("Too few confident mineral pixels to train on: check the SAM step output above.")

    pca = PCA(PCS, random_state=SEED).fit(cube.reshape(-1, b)[rng.choice(h * w, 50000, replace=False)])
    pcs_img = pca.transform(cube.reshape(-1, b)).reshape(h, w, PCS).astype(np.float32)
    pcs_img = (pcs_img - pcs_img.mean((0, 1))) / (pcs_img.std((0, 1)) + 1e-6)
    print(f"   PCA: {PCS} components keep {pca.explained_variance_ratio_.sum() * 100:.1f}% of the variance")

    train_m, test_m = spatial_split(h, w)
    tr_r, tr_c = np.where(train_m & (y_img >= 0))
    te_r, te_c = np.where(test_m & (y_img >= 0))
    sel = np.concatenate([rng.permutation(np.where(y_img[tr_r, tr_c] == k)[0])[:MAX_PER_CLASS] for k in range(len(classes))])
    tr_r, tr_c = tr_r[sel], tr_c[sel]
    ytr, yte = y_img[tr_r, tr_c], y_img[te_r, te_c]
    print(f"[2] Spatial split: {len(ytr):,} training pixels, {len(yte):,} test pixels (held-out blocks + {BUFFER}px buffer)")
    absent = [classes[k] for k in range(len(classes)) if not (yte == k).any()]
    if absent:
        print("   not present in the test blocks:", absent)

    print("[3] Random Forest on the 188-band spectrum (baseline)...")
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, class_weight="balanced")
    rf.fit(cube[tr_r, tr_c], ytr)
    p_rf = rf.predict(cube[te_r, te_c])

    print("[4] Random Forest with a RANDOM pixel split (for comparison only)...")
    all_r, all_c = np.where(y_img >= 0)
    a_tr, a_te = train_test_split(np.arange(len(all_r)), test_size=0.3, random_state=SEED, stratify=y_img[all_r, all_c])
    a_tr = a_tr[:len(ytr)]
    rf2 = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, class_weight="balanced")
    rf2.fit(cube[all_r[a_tr], all_c[a_tr]], y_img[all_r[a_tr], all_c[a_tr]])
    p_rand = rf2.predict(cube[all_r[a_te], all_c[a_te]])

    print("[5] 3D-CNN (spectral-spatial)...")
    net, hist = train_cnn(patches(pcs_img, tr_r, tr_c), ytr, len(classes))
    p_cnn = predict_at(net, pcs_img, te_r, te_c)

    results = [scores("3D-CNN (spatial split)", yte, p_cnn, classes),
               scores("Random Forest (spatial split)", yte, p_rf, classes),
               scores("Random Forest (random pixel split - inflated)", y_img[all_r[a_te], all_c[a_te]], p_rand, classes)]
    per_class = pd.DataFrame({"mineral": classes,
                              "test_pixels": [int((yte == k).sum()) for k in range(len(classes))],
                              "f1_3dcnn": f1_score(yte, p_cnn, labels=range(len(classes)), average=None, zero_division=0).round(3),
                              "f1_random_forest": f1_score(yte, p_rf, labels=range(len(classes)), average=None, zero_division=0).round(3)})
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS / "model_comparison.csv", index=False)
    per_class.to_csv(RESULTS / "per_class_f1.csv", index=False)
    import torch
    torch.save({"state_dict": net.state_dict(), "classes": classes, "pca_components": pca.components_,
                "pca_mean": pca.mean_, "patch": PATCH}, MODELS / "mineral_3dcnn.pt")

    print("[6] Full-scene 3D-CNN map...")
    all_rr, all_cc = np.divmod(np.arange(h * w), w)
    full = np.full(h * w, -1)
    valid = cube.reshape(-1, b).mean(1) > 0.02
    full[valid] = predict_at(net, pcs_img, all_rr[valid], all_cc[valid])
    full = full.reshape(h, w)
    np.save(RESULTS / "cnn_prediction_map.npy", full)
    figures(y_img, full, test_m, yte, p_cnn, classes, hist)

    summary = {"labels": f"SAM against the USGS library, angle <= {LABEL_ANGLE} rad (reference map, not field truth)",
               "split": f"spatial: {len(TEST_BLOCKS)} of {GRID * GRID} blocks held out, {BUFFER}-pixel buffer",
               "classes": classes, "train_pixels": int(len(ytr)), "test_pixels": int(len(yte)),
               "pca_components": PCS, "patch": PATCH, "epochs": EPOCHS, "results": results}
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nResults (agreement with the SAM / USGS reference on held-out areas):")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nPer-class F1:\n" + per_class.to_string(index=False))


def figures(y_img, full, test_m, yte, p_cnn, classes, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    n = len(classes)
    cmap = ListedColormap(["#000000"] + list(plt.get_cmap("tab20").colors[:n]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for a, img, t in ((ax[0], y_img, "Reference (SAM, confident pixels)"), (ax[1], full, "3D-CNN prediction")):
        im = a.imshow(img + 1, cmap=cmap, vmin=0, vmax=n, interpolation="nearest")
        a.contour(test_m, levels=[0.5], colors="white", linewidths=0.8)
        a.set_title(t); a.axis("off")
    cb = fig.colorbar(im, ax=ax[1], ticks=np.arange(n + 1), fraction=0.046)
    cb.ax.set_yticklabels(["None"] + classes, fontsize=7)
    fig.suptitle("White outlines: held-out test blocks")
    fig.tight_layout(); fig.savefig(RESULTS / "03_reference_vs_3dcnn_map.png", dpi=130); plt.close(fig)

    cm = confusion_matrix(yte, p_cnn, labels=range(n))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    ax.set_xticks(range(n), classes, rotation=60, fontsize=7); ax.set_yticks(range(n), classes, fontsize=7)
    ax.set(xlabel="3D-CNN", ylabel="Reference", title="Confusion matrix (held-out blocks)")
    fig.tight_layout(); fig.savefig(RESULTS / "04_confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(hist) + 1), hist, marker="o")
    ax.set(xlabel="Epoch", ylabel="Training loss", title="3D-CNN training")
    fig.tight_layout(); fig.savefig(RESULTS / "05_training_loss.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
