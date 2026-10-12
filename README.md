# Mineral Identification from Hyperspectral Satellite Data

**Domain:** Hyperspectral imaging – geology / mining exploration
**Study area:** Cuprite mining district, Nevada (USA), the standard test site for hyperspectral mineral mapping
**Methods:** band selection, USGS spectral library matching, VCA endmember extraction, Spectral Angle Mapper (SAM), 3D-CNN, Random Forest baseline

---

## What it does

1. **Loads AVIRIS reflectance** (NASA/JPL, 19 June 1997, scene `f970619t01p02_r02_sc03`, 614 × 512 pixels, 224 bands, 370–2500 nm).
2. **Removes 36 water-vapour and noisy bands** (224 → 188 bands).
3. **Builds a reference library** of 10 Cuprite minerals from the **USGS Spectral Library v7** (83 laboratory spectra), resampled to the AVIRIS bands with Gaussian band responses.
4. **Continuum removal in the SWIR (2080–2380 nm)**, where Al-OH, NH4, Fe/Mg-OH and carbonate absorptions identify these minerals. Brightness and iron slopes are removed, so only the absorption shape is compared.
5. **VCA endmember extraction**: finds the purest pixels in the scene and identifies each one against the library.
6. **SAM mineral map**: every pixel gets the closest mineral by spectral angle; weak or unmatched pixels stay "unclassified".
7. **3D-CNN** (spectral-spatial; 9 × 9 patches of 30 PCA components) trained on the confident SAM pixels and compared with a **Random Forest** on the 188-band spectrum.
8. **Honest evaluation with a spatial split**: 4 of 16 image blocks are held out, with a 4-pixel buffer, so test pixels and their neighbourhoods are never seen in training. A random pixel split is also reported to show how much it inflates accuracy.

## Results

**Mineral map (SAM, continuum-removed SWIR):** muscovite, calcite, alunite, jarosite, chalcedony, kaolinite, dickite and halloysite, matching the minerals known at Cuprite. 60% of pixels have no diagnostic SWIR absorption and are left unclassified (surfaces without a clear SWIR absorption).

**Classifiers (agreement with the SAM / USGS reference on held-out image blocks):**

| Model | Split | Overall accuracy | Kappa | Macro F1 |
|---|---|---|---|---|
| **3D-CNN (spectral-spatial)** | spatial | **94.3%** | **0.913** | 0.562 |
| Random Forest (188 bands) | spatial | 84.1% | 0.768 | 0.459 |
| Random Forest (188 bands) | random pixels | 92.8% | 0.895 | 0.571 |

On **all** SAM-mapped pixels in the held-out blocks (not only the confident training-quality ones), the 3D-CNN agrees with the SAM map on **89.6%**.

On **all** SAM-mapped pixels of the held-out blocks (not only the confident training-quality ones) the 3D-CNN agrees with SAM on **89.6%**.

**Per-mineral F1 (3D-CNN):** calcite 0.99, muscovite 0.97, alunite 0.92, chalcedony 0.80, jarosite 0.65, dickite 0.29, halloysite 0.23, kaolinite 0.21, montmorillonite 0.00 (only 18 test pixels).

**Ground-truth validation (Cuprite benchmark, 12 true mineral spectra):** VCA endmembers were matched to the ground-truth spectra (Hungarian matching, 10 runs).

| Mineral | Mean SAD (rad) |
|---|---|
| Montmorillonite | 0.066 |
| Kaolinite (2) | 0.080 |
| Andradite | 0.080 |
| Nontronite | 0.085 |
| Muscovite | 0.090 |
| Kaolinite (1) | 0.092 |
| Alunite | 0.093 |
| Buddingtonite | 0.093 |
| Dumortierite | 0.175 |
| Sphene | 0.434 |
| Pyrope | 0.615 |
| Chalcedony | 0.768 |

9 of the 12 minerals are recovered with a spectral angle below 0.1 rad (about 5 degrees), in line with published VCA results on this benchmark. Pyrope, sphene and chalcedony cover very few pixels, so VCA often misses them (mean SAD 0.22 rad over all 12; best run 0.15 rad). VCA is random, so results are averaged over 10 runs.

**Key findings**
- The 3D-CNN beats the spectrum-only Random Forest by 10 points on unseen areas because it also uses the spatial neighbourhood.
- A random pixel split makes the Random Forest look 8.7 points better than it really is (92.8% vs 84.1%), because neighbouring, nearly identical pixels end up in both training and test sets. Spatial splits should be used for hyperspectral images.
- Rare minerals (fewer than 200 test pixels) are classified poorly; more labelled pixels or class-balanced sampling would be needed.

## Figures (`results/figures/`)

| File | Content |
|---|---|
| `01_sam_mineral_map.png` | True-colour image and SAM mineral map |
| `02_endmember_signatures.png` | VCA endmember spectra with their USGS match |
| `03_reference_vs_3dcnn_map.png` | Reference map vs 3D-CNN map, held-out blocks outlined |
| `04_confusion_matrix.png` | 3D-CNN confusion matrix on the held-out blocks |
| `05_training_loss.png` | 3D-CNN training loss |
| `06_benchmark_endmembers.png` | Benchmark: VCA endmembers vs ground-truth spectra |

Tables (`results/tables/`): `metrics.json`, `model_comparison.csv`, `per_class_f1.csv`, `endmembers_vca.csv`, `sam_mineral_distribution.csv`.
Full-scene maps (`results/maps/`): SAM and 3D-CNN class maps as NumPy arrays.

## How to run

Double-click **`run.bat`**, or:

```bash
pip install -r requirements.txt
python -m pipeline.run_all
```

| Step | Script | Output |
|---|---|---|
| 1 | `pipeline/s01_spectral_analysis.py` | 188-band cube, library, VCA endmembers, SAM map |
| 2 | `pipeline/s02_train_3dcnn.py` | 3D-CNN and Random Forest, metrics, maps |
| 3 | `pipeline/s03_figures.py` | presentation figures (one colour per mineral) |

Settings (bands, SWIR range, angle thresholds, minerals) are in `pipeline/common.py`.

## Project structure

```
Mineral_Identification/
├── Datasets/                      ALL data in one place
│   ├── AVIRIS_Cuprite/            AVIRIS 1997 Cuprite reflectance (+ .spc spectral calibration)
│   ├── USGS_Spectral_Library/     USGS Spectral Library v7 (splib07a)
│   ├── Sentinel2/                 Sentinel-2 band (kept for reference)
│   ├── Cuprite_Benchmark/         standard benchmark: 188-band subimage + 12 ground-truth mineral spectra
│   └── processed/                 188-band cube, library, SAM labels (generated)
├── pipeline/                      all code (s01 spectral analysis, s02 3D-CNN, s03 figures)
├── models/mineral_3dcnn.pt        trained 3D-CNN (with PCA parameters)
├── results/                       metrics and figures
├── _backups/                      earlier results and models, kept for reference
├── README.md
└── requirements.txt
```

## Data sources

- **AVIRIS:** NASA/JPL Airborne Visible/Infrared Imaging Spectrometer, Cuprite flight line f970619t01p02_r02 (reflectance product).
- **USGS Spectral Library Version 7** (Kokaly et al., 2017), splib07a.

## Limitations

- The classifier labels come from SAM against laboratory spectra, not from field samples, so classification accuracies measure agreement with this physics-based reference. The endmember extraction itself is checked against ground truth (Cuprite benchmark); a field-verified mineral map (for example the USGS Tetracorder Cuprite map) would be the next step for the classifiers.
- The 1997 reflectance product has atmospheric artefacts below 2060 nm and above 2400 nm; those bands are left out of the SWIR analysis.
- Buddingtonite (a key Cuprite mineral) has only 2 library spectra and was not detected with the current thresholds.
- One scene only; other scenes or sensors (AVIRIS-NG, PRISMA, EnMAP) would test generalisation.

## References

- Clark, R. N. & Roush, T. L. (1984). Reflectance spectroscopy: quantitative analysis techniques for remote sensing applications. *JGR*.
- Kruse, F. A. et al. (1993). The Spectral Image Processing System (SIPS): interactive visualization and analysis of imaging spectrometer data (SAM). *Remote Sensing of Environment*.
- Nascimento, J. M. P. & Bioucas-Dias, J. M. (2005). Vertex Component Analysis. *IEEE TGRS*.
- Kokaly, R. F. et al. (2017). USGS Spectral Library Version 7. *USGS Data Series 1035*.
- Roy, S. K. et al. (2020). HybridSN: exploring 3-D–2-D CNN feature hierarchy for hyperspectral image classification. *IEEE GRSL*.
