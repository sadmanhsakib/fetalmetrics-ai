# 🩺 fetalmetrics-ai

**Automated Fetal Biometry Assistant** — an end-to-end system that measures
fetal **head circumference (HC)** from a 2D ultrasound image, places it on a
gestational-age growth curve, and reports a screening risk band.

> ⚠️ **Research prototype — not a certified diagnostic device.** Outputs are for
> engineering demonstration and research only and must not be used for clinical
> decision-making.

Given an ultrasound and a gestational age, the app segments the fetal skull with
one of two exported deep-learning models (**YOLOv8-seg** or **U-Net**), fits an
ellipse to the skull, converts the perimeter to millimetres, and classifies the
growth percentile against a **Hadlock** reference.

---

## ✨ Features

- **Dual architecture, swap in real time** — YOLOv8-seg and U-Net served side by
  side via ONNX Runtime (no PyTorch/TensorFlow needed at inference).
- **Calibrated measurement** — pixel→mm scaling from **HC18** metadata, with a
  manual override for arbitrary images.
- **Ramanujan HC** — ellipse perimeter via Ramanujan's II approximation, the same
  convention used to derive HC18 ground truth.
- **Clinical read-out** — HC (mm), estimated percentile, z-score, and a
  colour-coded risk band with a zoned percentile gauge.
- **Honest by design** — an auditable, swappable reference table; graceful
  behaviour when weights are absent; a prominent research-only disclaimer.
- **Tested core** — the geometry and clinical maths are unit-tested against known
  ground truth, and the ONNX decode paths are tested with synthetic tensors.

---

## 🖼️ Interface

A dark clinical "research console": the grayscale ultrasound and the cyan
segmentation overlay read clearly on a deep ink canvas; the three risk colours
(crimson / amber / emerald) are reserved strictly for clinical meaning.

```
🩺 fetalmetrics-ai — Automated Fetal Biometry Assistant
⚠️  Research prototype — not for clinical diagnosis.
────────────────────────────────────────────────────────────
[ Original ultrasound ]        [ Segmentation overlay ]
────────────────────────────────────────────────────────────
 HC 266.5 mm      Percentile 19.2th      🟡 MEDIUM RISK
 ▇▇▇▇▇▇▇▇▇▇░░░░░░░░░░░░░░░░░░░░  (10th · 25th · 50th · 90th)
 Inference 142 ms · Post 9 ms · 0.20 mm/px (HC18) · conf 96.3%
```

Regenerate a static preview any time with `python build_preview.py` →
`preview.html`.

---

## 🔬 How it works

```
Ultrasound ─▶ Segmentation (YOLOv8-seg | U-Net, ONNX)
           ─▶ Largest contour ─▶ cv2.fitEllipse
           ─▶ semi-axes px → mm  (HC18 pixel size / manual)
           ─▶ HC = Ramanujan perimeter
           ─▶ z = (HC − mean_GA)/SD_GA  (Hadlock reference)
           ─▶ percentile = Φ(z)  ─▶  risk band
```

**Head circumference (Ramanujan II):**

```
h = ((a − b) / (a + b))²
C ≈ π (a + b) · [ 1 + 3h / (10 + √(4 − 3h)) ]
```
where `a`, `b` are the ellipse semi-axes in millimetres.

**Screening bands (percentile):** `< 10th` → **High** (possible IUGR) · `10–25th`
→ **Medium** (borderline) · `> 25th` → **Normal**.

---

## 🚀 Quickstart

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt

# 2. add your exported models  (see below)
#    models/yolov8_hc.onnx
#    models/unet_hc.onnx

# 3. (optional) add HC18 calibration CSVs to data/

# 4. run
streamlit run app.py
```

The app launches with or without weights — without them you still get the full
UI and clear guidance on where to drop the `.onnx` files.

---

## 📦 Model weights (ONNX)

Export your trained networks to ONNX and place them here:

| Model       | Expected path             | Family in `config.py` |
|-------------|---------------------------|-----------------------|
| YOLOv8-seg  | `models/yolov8_hc.onnx`   | `yolov8_seg`          |
| U-Net       | `models/unet_hc.onnx`     | `unet`                |

Everything the inference layer needs is declared in **`config.py → MODELS`**.
Adjust these to match how *you* exported each network — they are the only values
you should ever need to touch:

```python
ModelSpec(
    input_size=(640, 640),   # (H, W) fed to the network
    channels=3,              # 1 = grayscale, 3 = RGB
    normalize="scale",       # "scale" (/255) or "standard" (mean/std)
    conf_threshold=0.25,     # YOLOv8 only
)
```

**Expected export shapes**

- *YOLOv8-seg*: `output0 (1, 4+nc+32, num_anchors)` + `output1 (1, 32, mh, mw)`.
  Number of classes `nc` is inferred automatically, so a single-class "fetal
  head" model works unchanged.
- *U-Net*: `(1, 1, H, W)` / `(1, H, W)` (logits or probabilities), or
  `(1, 2, H, W)` two-class logits. Sigmoid/softmax is applied automatically when
  the output is not already in `[0, 1]`.

If a run fails, the app surfaces the exact error and points you at the config —
the usual cause is an input-size or normalization mismatch.

---

## 📐 Calibration (HC18)

The [HC18 Grand Challenge](https://hc18.grand-challenge.org/) ships CSVs mapping
each image filename to its `pixel size(mm)`. Drop either file into `data/`:

```
data/training_set_pixel_size_and_HC.csv
data/test_set_pixel_size.csv
```

When an uploaded filename matches, its scale is used automatically. Otherwise
enter **mm per pixel** manually in the sidebar (or tick *Force manual scale* to
override HC18). Column names are matched tolerantly (`filename`, `pixel size(mm)`).

---

## 🧮 Clinical reference

Percentiles use a **Hadlock** HC-for-gestational-age reference — mean and SD per
week — living in `src/clinical/reference_hadlock.py` as a small, transparent,
**swappable** table. To use your own calibrated coefficients or an
**INTERGROWTH-21st** table, edit that one file; nothing else changes.

> The bundled coefficients are literature-consistent and intended for
> demonstration. Verify them against your own fit before any real use.

---

## 🗂️ Project structure

```
fetalmetrics-ai/
├── app.py                     # Streamlit UI + orchestration
├── config.py                  # ALL tunables (paths, models, colours, bands)
├── requirements.txt
├── .streamlit/config.toml     # theme
├── assets/styles.css          # clinical console stylesheet
├── src/
│   ├── inference/             # ONNX layer
│   │   ├── base.py            #   session mgmt + shared numpy ops
│   │   ├── yolov8_onnx.py     #   YOLOv8-seg decode
│   │   ├── unet_onnx.py       #   U-Net decode
│   │   └── registry.py        #   model factory + availability
│   ├── postprocess/
│   │   ├── ellipse.py         # contour → fitEllipse → Ramanujan HC
│   │   └── overlay.py         # cyan ellipse + crosshair renderer
│   ├── clinical/
│   │   ├── reference_hadlock.py   # HC-for-GA mean/SD (swappable)
│   │   ├── percentiles.py         # z-score + normal-CDF percentile
│   │   └── risk.py                # 3-band classification + colours
│   ├── calibration/pixel_size.py  # HC18 lookup + manual override
│   └── ui/                    # theme + HTML components
├── models/  data/  samples/   # weights, HC18 CSVs, demo images
└── tests/                     # pipeline + inference-decode tests
```

---

## ✅ Testing

```bash
python tests/test_pipeline.py          # geometry + clinical maths vs ground truth
python tests/test_inference_decode.py  # ONNX decode on synthetic tensors
# or, if you have pytest:
pytest -q
```

`test_pipeline` verifies the Ramanujan perimeter against a circle's exact value,
recovers a synthetic ellipse's HC to <2%, and checks percentile/risk banding.
`test_inference_decode` drives both wrappers with fake ONNX outputs so the decode
plumbing is validated without the trained weights.

---

## ☁️ Deployment

Runs on CPU ONNX Runtime, so it deploys cleanly to **Streamlit Community Cloud**
or a **Hugging Face Space** (Streamlit SDK): push the repo, ensure the `.onnx`
files are present (Git LFS or a download step), and set `app.py` as the entry
point. Increase upload size in `.streamlit/config.toml` if needed.

---

## 📚 References

- Hadlock FP, Deter RL, Harrist RB, Park SK. *Estimating fetal age: computer-
  assisted analysis of multiple fetal growth parameters.* Radiology 1984;152:497–501.
- van den Heuvel TLA, de Bruijn D, de Korte CL, van Ginneken B. *Automated
  measurement of fetal head circumference using 2D ultrasound images (HC18).*
  PLoS ONE 2018.
- Ramanujan S. *Modular equations and approximations to π.* Quarterly Journal of
  Mathematics 1914.
- Jocher G. et al. *Ultralytics YOLOv8.* 2023.
- Ronneberger O, Fischer P, Brox T. *U-Net: Convolutional Networks for
  Biomedical Image Segmentation.* MICCAI 2015.

---

## ⚖️ Disclaimer

This software is a research and educational prototype. It is **not** a medical
device, has not been clinically validated, and must not be used for diagnosis or
any patient-care decision.
