# Models

Everything in this repository, code and model weights alike, is released under the **MIT licence** (see
`LICENSE`). The weights were trained on public NASA TESS data and on public TOI / TFOPWG dispositions; they carry
the attribution in [Data sources and attribution](#data-sources-and-attribution).

These models are experimental. They have not been reviewed by an astrophysicist (see the
[Confidence Ledger](CONFIDENCE_LEDGER.md)); their scores are triage signals, not detections.

## Shipped weights

| File | Task | Inputs | Training data (summary) | Date | Size |
|---|---|---|---|---|---|
| `worker/models/transit_cnn.onnx` (+ `.onnx.data`) and the identical copy `exonit/models/transit_cnn.onnx` | First-generation transit classifier: planet-like vs not, on a phase-folded curve | `phase_folded_flux` `[batch, 1, 200]` | Original small training set (kept for the legacy path and for tests) | 2026 | 24 KB + 43 KB |
| `worker/models/cnn_ffi_v1.onnx` + `worker/models/cnn_ffi_v1.onnx.data` | CNN-FFI: planet-like transit vs artefact / eclipsing binary / no transit, per candidate | `global_view` `[batch, 1, 201]`, `local_view` `[batch, 1, 61]` | FFI light curves (TGLC, QLP) of TOI dispositions plus synthetic injections and hard-negative families; TIC-blocked train/validation/test split | 2026 | 12 KB + 2.3 MB |
| `worker/models/exominer_ffi_v6_ensemble.onnx` + `worker/models/exominer_ffi_v6_ensemble.onnx.data` | ExoMiner-FFI v6: multi-branch candidate scorer for FFI-only stars (no SPOC Data Validation product) | `global_view` 201, `local_view` 61, `centroid_view` 61 with `centroid_mask`, `stellar` 5, `tce` 4, `aux_views` 3 x 61 | TESS FFI light curves with TOI/TFOPWG dispositions and synthetic injections, ensemble of seeds | 2026 | 220 KB + 3.6 MB |
| `worker/models/exominer_ffi_v6_ensemble.calib.json` | Piecewise-linear calibration of the v6 ensemble output (maps the raw score at 85% recall to 0.5) | the raw ensemble output | Calibrated on a held-out slice | 2026 | 4 KB |
| `models/verdict_classifier_v2_1_2026-05-01.pkl` | Verdict v2.1 (stars with a SPOC Data Validation product): three classes `likely_false_positive`, `uncertain`, `planet_candidate` | candidate features: log period, log depth, duration, SNR, CNN score, FPP, ExoMiner score, number of sectors, time span, and others | 7,587 labels over 2,472 TICs (SPOC-path dispositions) | 2026-05-01 | 6.8 MB |
| `models/verdict_classifier_v3_tfopwg_2026-09-19_nofpp.pkl` | Verdict v3 TFOPWG, variant without the FPP feature: the FFI-path verdict, trained against TFOPWG dispositions | candidate features without TRICERATOPS FPP; known-host catalogue features are excluded on purpose (label leakage) | TOI / TFOPWG dispositions | 2026-09-19 | 336 KB |

`transit_cnn.onnx` is loaded by `worker/cnn.py` (200-bin phase-folded flux, output `planet_probability`);
`cnn_ffi_v1` by `worker/cnn_ffi.py`; `exominer_ffi_v6_ensemble` by `worker/exominer_ffi_inference.py` (model version
string written to each candidate: `phase3_v6_ensemble_cal`). The `.onnx.data` files hold the external tensor data of
the matching `.onnx` file and must stay next to it.

### Server settings that select the verdict models

The public server must run with:

```
EXONIT_VERDICT_VERSION=v2.1
EXONIT_VERDICT_FFI_VERSION=v3_tfopwg
```

An earlier v1 verdict model exists in the project history but is not shipped; do not set the variables to `v1`.

### ExoMiner-FFI is not NASA's ExoMiner

"ExoMiner-FFI" is an in-house ONNX ensemble trained on TESS full-frame-image light curves, using the published
ExoMiner multi-branch design (global, local, centroid, stellar and TCE inputs) as the architectural reference.
It is **not** the NASA ExoMiner weights. The NASA container can optionally be run server-side for stars with an
official SPOC DV product (`docs/advanced/exominer-docker.md`), but it is not part of this repository and is off
by default.

### Superseded file in the worker tree

`worker/models/exominer_ffi_v1.onnx` (+ `.onnx.data`) is the first ExoMiner-FFI model, superseded by v6. Nothing
loads it; it is present only because the worker tree is exported as a whole.

### Not shipped

Larger research checkpoints, intermediate training sets and unreleased candidates (later CNN and ExoMiner
versions) are intentionally not distributed.

## Data sources and attribution

- **NASA TESS** mission data, archived at MAST (Mikulski Archive for Space Telescopes). Public domain data;
  please acknowledge TESS and MAST in any derived work.
- **TGLC** (TESS-Gaia Light Curves), a High Level Science Product on MAST derived from TESS full-frame images and
  Gaia DR3 (Han and Brandt). Cite the TGLC paper if you use these light curves.
- **TOI catalogue and TFOPWG dispositions**, from the NASA Exoplanet Archive / ExoFOP-TESS (the TESS Objects of
  Interest table and its follow-up dispositions), used as training labels.
- **QLP** (Quick-Look Pipeline) and **SPOC** light curves, MAST High Level Science Products.
- **Gaia DR3** and the **TESS Input Catalog** for stellar parameters and cross-matching.
- Architecture references: AstroNet (Shallue and Vanderburg 2018; Yu et al. 2019) for the CNN, ExoMiner
  (Valizadegan et al. 2022) for the multi-branch scorer, TRICERATOPS (Giacalone et al. 2021) for false-positive
  probabilities.

The weights are released under the MIT licence as above; the underlying data remain subject to their own terms.
