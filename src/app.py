"""
app.py — fetalmetrics-ai
========================
Streamlit entry point. Orchestrates: image upload → calibration → ONNX
segmentation (YOLOv8-seg / U-Net) → ellipse fit → Ramanujan HC → Hadlock
percentile → risk classification, with a clinical research-console UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st
from PIL import Image

import config
from calibration import pixel_size as calib
from clinical import percentiles as pct
from clinical import reference_hadlock as ref
from clinical import risk as rk
from inference import registry
from postprocess import ellipse as geo
from postprocess import overlay as ov
from ui import components as C
from ui import theme


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_segmenter(model_key: str):
    """One ONNX session per model per process."""
    return registry.build_segmenter(model_key)


def load_image(uploaded) -> np.ndarray:
    """Uploaded file -> RGB uint8 numpy array."""
    return np.array(Image.open(uploaded).convert("RGB"))


# --------------------------------------------------------------------------- #
# Sidebar — input controls
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<div class="fm-side-brand"><b>🩺 Fetal <span class="d">Metrics-AI</span></b></div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown(C.section_label("1 · Ultrasound image"), unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a 2D fetal head ultrasound",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        st.markdown(C.section_label("2 · Gestational age"), unsafe_allow_html=True)
        ga_weeks = st.slider(
            "Gestational age (weeks)",
            min_value=config.GA_MIN_WEEKS, max_value=config.GA_MAX_WEEKS,
            value=config.GA_DEFAULT_WEEKS, step=config.GA_STEP_WEEKS,
            help="Used to place the measured HC on the Hadlock growth curve.",
        )

        st.markdown(C.section_label("3 · Segmentation model"), unsafe_allow_html=True)
        statuses = {s.key: s for s in registry.all_statuses()}
        labels = {config.MODELS[k].display_name: k for k in config.MODEL_ORDER}
        choice = st.radio(
            "Model", list(labels.keys()), index=0,
            label_visibility="collapsed", horizontal=True,
        )
        model_key = labels[choice]

        # availability chips
        chips = []
        for k in config.MODEL_ORDER:
            s = statuses[k]
            dot = "on" if s.available else "off"
            state = "loaded" if s.available else "missing weights"
            chips.append(
                f'<span class="fm-chip"><span class="d {dot}"></span>'
                f'{s.display_name}: {state}</span>'
            )
        st.markdown(" ".join(chips), unsafe_allow_html=True)
        st.caption(config.MODELS[model_key].description)

        # calibration
        st.markdown(C.section_label("4 · Calibration"), unsafe_allow_html=True)
        auto_hint = ""
        default_val = float(config.DEFAULT_PIXEL_SIZE_MM)
        image_rgb = None
        if uploaded is not None:
            try:
                uploaded.seek(0)
                image_rgb = load_image(uploaded)
                hc18 = calib.lookup_hc18(uploaded.name, image_rgb)
            except Exception:
                hc18 = calib.lookup_hc18(uploaded.name)
            if hc18:
                auto_hint = f"HC18 metadata found: {hc18:.5f} mm/px"
                default_val = hc18
        if auto_hint:
            st.caption("✅ " + auto_hint)
        elif calib.hc18_available():
            st.caption("No HC18 match for this filename — use manual scale below.")
        else:
            st.caption("No HC18 CSV loaded — enter the pixel scale manually.")

        manual_mm_per_px = st.number_input(
            "Pixel scale (mm per pixel)",
            min_value=0.010, max_value=1.000,
            value=default_val, step=0.001, format="%.4f",
            help="Physical size of one pixel. Auto-filled from HC18 when available.",
        )
        prefer_manual = st.checkbox(
            "Force manual scale (override HC18)", value=False,
        )

    return {
        "uploaded": uploaded,
        "image_rgb": image_rgb,
        "ga_weeks": ga_weeks,
        "model_key": model_key,
        "manual_mm_per_px": manual_mm_per_px,
        "prefer_manual": prefer_manual,
    }


# --------------------------------------------------------------------------- #
# Main — results
# --------------------------------------------------------------------------- #
def render_dashboard(hc, percentile, risk, cal, seg, post_ms, ga_weeks) -> None:
    summary = ref.reference_summary(ga_weeks)
    delta = hc.hc_mm - summary.mean_mm
    delta_cls = "pos" if delta >= 0 else "neg"

    st.markdown(C.section_label("Clinical summary"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            C.metric_card_html(
                "Head circumference", f"{hc.hc_mm:.1f}", "mm",
                f'Expected {summary.mean_mm:.1f} mm · '
                f'<span class="{delta_cls}">Δ {delta:+.1f} mm</span>',
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            C.metric_card_html(
                "Estimated percentile", percentile.percentile_label, "",
                f"z-score {percentile.z_score:+.2f} · GA {ga_weeks:.1f} wk",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(C.risk_card_html(risk), unsafe_allow_html=True)

    st.markdown(C.section_label("Growth percentile"), unsafe_allow_html=True)
    st.markdown(C.gauge_html(percentile.percentile), unsafe_allow_html=True)

    conf = f"{seg.confidence*100:.1f}%" if seg.confidence is not None else "—"
    st.markdown(
        C.timing_strip_html([
            ("Inference", f"{seg.inference_ms:.0f} ms ({seg.model_name})"),
            ("Post-processing", f"{post_ms:.0f} ms"),
            ("Calibration", f"{cal.mm_per_px:.4f} mm/px · {cal.source_label}"),
            ("Model confidence", conf),
            ("Reference", "Hadlock HC-for-GA"),
        ]),
        unsafe_allow_html=True,
    )
    if not cal.is_trustworthy:
        st.warning(
            "Calibration is using a fallback default — the HC value is not "
            "physically calibrated. Provide HC18 metadata or a manual mm/px scale.",
            icon="⚠️",
        )


def render_missing_weights(model_key: str) -> None:
    spec = config.MODELS[model_key]
    st.markdown(C.section_label("Model not loaded"), unsafe_allow_html=True)
    st.info(
        f"**{spec.display_name}** weights were not found. Place the exported "
        f"ONNX file at:\n\n`{spec.weights_path}`\n\n"
        "Then reload. You can adjust the expected filename, input size and "
        "normalization in `config.py → MODELS`.",
        icon="📦",
    )


def render_results(state: dict) -> None:
    image = state["image_rgb"]
    if image is None:
        state["uploaded"].seek(0)
        image = load_image(state["uploaded"])

    cal = calib.resolve(
        filename=state["uploaded"].name,
        image_rgb=image,
        manual_mm_per_px=state["manual_mm_per_px"],
        prefer_manual=state["prefer_manual"],
    )

    segmenter = get_segmenter(state["model_key"])
    if not segmenter.available:
        # Still show the input image so the upload is confirmed.
        st.markdown(C.section_label("Input"), unsafe_allow_html=True)
        st.markdown(C.image_caption("Original ultrasound", state["uploaded"].name),
                    unsafe_allow_html=True)
        st.image(image, width='stretch')
        render_missing_weights(state["model_key"])
        return

    # -- run model -------------------------------------------------------- #
    try:
        with st.spinner(f"Segmenting with {segmenter.name}…"):
            seg = segmenter.predict(image)
    except Exception as exc:  # noqa: BLE001 — surface export mismatches to the user
        import traceback
        traceback.print_exc()
        st.error(
            f"Inference failed for {segmenter.name}: `{exc}`\n\n"
            "This usually means the ONNX input size / normalization / output "
            "layout differs from the defaults. Adjust `config.py → MODELS`.",
            icon="🛑",
        )
        return

    # -- post-process ----------------------------------------------------- #
    t0 = time.perf_counter()
    hc = geo.measure_hc(seg.mask, cal.mm_per_px)
    overlay_img = ov.render(image, hc, mask=seg.mask) if hc else image
    post_ms = (time.perf_counter() - t0) * 1000.0

    # -- images side by side --------------------------------------------- #
    st.markdown(C.section_label("Segmentation"), unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(C.image_caption("Original ultrasound", state["uploaded"].name),
                    unsafe_allow_html=True)
        st.image(image, width='stretch')
    with col_b:
        st.markdown(C.image_caption("Segmentation overlay", "ellipse + axes"),
                    unsafe_allow_html=True)
        st.image(overlay_img, width='stretch')

    if hc is None:
        st.warning(
            f"{segmenter.name} did not return a confident skull segmentation on "
            "this image. Try the other model, a clearer scan, or check the input "
            "size in config.py.",
            icon="🔍",
        )
        return

    # -- clinical readout ------------------------------------------------- #
    percentile = pct.evaluate(hc.hc_mm, state["ga_weeks"])
    risk = rk.classify(percentile.percentile)
    render_dashboard(hc, percentile, risk, cal, seg, post_ms, state["ga_weeks"])


# --------------------------------------------------------------------------- #
# Methodology (always available — signals the research depth)
# --------------------------------------------------------------------------- #
def render_methodology() -> None:
    with st.expander("Methodology, references & limitations"):
        st.markdown(
            """
**Pipeline.** The uploaded 2D ultrasound is segmented by one of two exported
architectures — a **YOLOv8-seg** single-stage detector with a prototype mask
head, or a **U-Net** encoder–decoder — both served via ONNX Runtime. The
predicted skull mask is cleaned, the largest contour is isolated, and a
least-squares **ellipse** is fitted (`cv2.fitEllipse`).

**Head circumference.** The ellipse semi-axes are converted from pixels to
millimetres using the image calibration (HC18 `pixel size(mm)` metadata, or a
manual scale). The circumference is the ellipse perimeter, computed with the
**Ramanujan II** approximation
`C ≈ π(a+b)[1 + 3h/(10 + √(4−3h))]`, `h = ((a−b)/(a+b))²` — the same convention
used to derive HC18 ground truth.

**Growth percentile.** The measured HC is placed on a **Hadlock**
head-circumference-for-gestational-age reference (mean and SD per week). The
z-score `(HC − mean)/SD` maps to a percentile via the normal CDF.

**Screening bands.** `<10th` percentile → **High** (possible IUGR); `10–25th` →
**Medium** (borderline); `>25th` → **Normal**.

**References.** Hadlock FP et al., *Radiology* 1984;152:497-501. van den Heuvel
TLA et al., *PLoS ONE* 2018 (HC18 Grand Challenge). Ramanujan S., *Quarterly
Journal of Mathematics* 1914.

**Limitations.** A research prototype: percentile reference coefficients should
be independently verified before any real use; segmentation quality depends on
image quality and on how closely the ONNX export matches the configured
pre/post-processing; and single-measurement percentiles cannot replace serial
clinical assessment. **Not a certified diagnostic device.**
            """
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    theme.configure_page()
    theme.inject_styles()

    st.markdown(C.header_html(), unsafe_allow_html=True)
    st.markdown(C.safety_banner_html(), unsafe_allow_html=True)

    state = render_sidebar()

    if state["uploaded"] is None:
        st.markdown(C.empty_state_html(), unsafe_allow_html=True)
        if not registry.any_available():
            st.info(
                "No model weights detected yet. Drop `yolov8_hc.onnx` and "
                "`unet_hc.onnx` into the `models/` folder to enable inference.",
                icon="📦",
            )
    else:
        render_results(state)

    render_methodology()


if __name__ == "__main__":
    main()
