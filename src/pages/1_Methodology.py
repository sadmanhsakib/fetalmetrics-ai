"""
pages/1_Methodology.py
======================
The methodology as its own page (Streamlit multipage). It is reachable from the
sidebar navigation and from the prominent "Methodology" link in the homepage
masthead / empty state.

The document is written in a medical-journal register and — importantly — pulls
its numbers directly from the live clinical modules (`reference_hadlock`,
`config`, `percentiles`). The Hadlock reference chart, the SD model, the risk
thresholds and the model export specs shown below therefore always match the
code that actually runs; editing a coefficient in one place updates the
methodology automatically.
"""

from __future__ import annotations

import config
from clinical import reference_hadlock as ref
from ui import components as C
from ui import theme

# --------------------------------------------------------------------------- #
# Page chrome
# --------------------------------------------------------------------------- #
theme.configure_page("methodology")
theme.inject_styles()
theme.render_sidebar_nav("methodology")

import streamlit as st  # noqa: E402  (after set_page_config, by design)


# --------------------------------------------------------------------------- #
# Values pulled live from the clinical layer
# --------------------------------------------------------------------------- #
yolo = config.MODELS["yolov8"]
unet = config.MODELS["unet"]
pp = config.POSTPROCESS
hi = config.HIGH_RISK_MAX_PCT
md = config.MEDIUM_RISK_MAX_PCT
rc = config.RISK_COLORS

ga_lo, ga_hi = int(ref.GA_MIN_WEEKS), int(ref.GA_MAX_WEEKS)
sd_lo, sd_hi = ref.sd_hc_mm(ga_lo), ref.sd_hc_mm(ga_hi)

# Hadlock reference chart (integer completed weeks) — generated from the code.
_ref_rows = []
for wk in range(ga_lo, ga_hi + 1):
    s = ref.reference_summary(wk)
    _ref_rows.append(
        f"<tr><td>{wk}</td><td>{s.mean_mm:.0f}</td><td>{s.sd_mm:.1f}</td>"
        f"<td>{s.p10_mm:.0f}</td><td>{s.p50_mm:.0f}</td><td>{s.p90_mm:.0f}</td></tr>"
    )
ref_rows_html = "".join(_ref_rows)

# Risk-band rows, coloured by the same tokens the app uses.
bands = [
    (rc["HIGH"]["solid"], f"&lt; {hi:.0f}th", "High — screening alert",
     "Possible fetal growth restriction (IUGR). Flag for clinical correlation; "
     "a single low measurement is a prompt to review, not a diagnosis."),
    (rc["MEDIUM"]["solid"], f"{hi:.0f}–{md:.0f}th", "Borderline",
     "Lower-normal growth range. Monitor closely on serial scans rather than "
     "acting on the isolated value."),
    (rc["NORMAL"]["solid"], f"&ge; {md:.0f}th", "Normal",
     "Within the expected head-circumference growth curve for gestational age."),
]
bands_html = "".join(
    f'<div class="doc-band" style="--rk:{c};">'
    f'<div class="tag">{tag}</div>'
    f'<div class="bd"><b>{title}</b><p>{body}</p></div></div>'
    for c, tag, title, body in bands
)


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
DOC = f"""
<div class="doc">

  <a class="back-link" href="/" target="_self">{C.icon('arrow_left')}<span>Back to analyzer</span></a>

  <div class="doc-hero">
    <div class="eyebrow">Methods &amp; Clinical Reference</div>
    <h1>How {config.APP_NAME.split('-')[0].strip()}-AI measures fetal head circumference</h1>
    <p class="lede">A transparent, end-to-end account of the measurement chain — from raw
    ultrasound to a calibrated head circumference and its growth percentile — including the
    exact formulae, the reference distribution, the screening logic, and the assumptions and
    limitations that bound how the output may be read.</p>
    <div class="doc-meta">
      <span><b>Version</b> {config.APP_VERSION}</span>
      <span><b>Reference</b> Hadlock HC-for-GA</span>
      <span><b>Supported GA</b> {ga_lo}–{ga_hi} weeks</span>
      <span><b>Status</b> Research prototype</span>
    </div>
  </div>

  <div class="doc-toc">
    <div class="t">Contents</div>
    <ol>
      <li><a href="#overview">Overview &amp; intent</a></li>
      <li><a href="#pipeline">The measurement pipeline</a></li>
      <li><a href="#calibration">Spatial calibration</a></li>
      <li><a href="#segmentation">Skull segmentation</a></li>
      <li><a href="#ellipse">Contour isolation &amp; ellipse fit</a></li>
      <li><a href="#circumference">Head circumference</a></li>
      <li><a href="#percentile">Growth percentile</a></li>
      <li><a href="#risk">Screening risk bands</a></li>
      <li><a href="#config">Reproducibility &amp; configuration</a></li>
      <li><a href="#limitations">Assumptions &amp; limitations</a></li>
    </ol>
  </div>

  <div class="doc-body">

    <section class="doc-sec" id="overview">
      <div class="kicker">§ 1</div>
      <h2>Overview &amp; intent</h2>
      <p>{config.APP_NAME.split('-')[0].strip()}-AI estimates the fetal <strong>head circumference
      (HC)</strong> from a single two-dimensional ultrasound of the fetal head, and places that
      measurement on a gestational-age growth curve to produce a percentile and a screening
      band. The goal is a fully <em>auditable</em> pipeline: every step is a small, inspectable
      transform, and every clinical constant lives in one configuration file.</p>
      <p>The measurement problem is fundamentally geometric. An ultrasound of the fetal skull
      at the standard trans-thalamic plane shows an approximately elliptical bony outline; HC is
      the perimeter of the ellipse that best fits that outline. The engineering task is therefore
      to (a) find the skull, (b) fit an ellipse to it, (c) convert pixels to millimetres, and
      (d) compare the result against a normative reference. Each is treated below.</p>
      <div class="doc-callout" style="--rk:{rc['MEDIUM']['solid']};">
        <div class="h">{C.icon('warning')} Research prototype — not a diagnostic device</div>
        <p>{config.SAFETY_NOTICE}</p>
      </div>
    </section>

    <section class="doc-sec" id="pipeline">
      <div class="kicker">§ 2</div>
      <h2>The measurement pipeline</h2>
      <p>A single upload flows through five deterministic stages. Stages 1 and 2 depend on the
      input image and the chosen model; stages 3–5 are pure geometry and statistics and are
      unit-testable in isolation.</p>
      <ol class="doc-steps">
        <li><b>Calibrate.</b> Resolve the physical scale of the image in millimetres per pixel.</li>
        <li><b>Segment.</b> Predict a binary skull mask with an exported ONNX model (YOLOv8-seg or U-Net).</li>
        <li><b>Fit.</b> Clean the mask, isolate the largest contour, and fit an ellipse by least squares.</li>
        <li><b>Measure.</b> Convert the ellipse semi-axes to millimetres and compute its perimeter (HC).</li>
        <li><b>Interpret.</b> Convert HC + gestational age to a z-score, a percentile, and a screening band.</li>
      </ol>
    </section>

    <section class="doc-sec" id="calibration">
      <div class="kicker">§ 3</div>
      <h2>Spatial calibration (pixels → millimetres)</h2>
      <p>No measurement is meaningful without a physical scale. Every pixel spacing is resolved to
      a single number, <code>mm_per_px</code>, from the first available source in this priority
      order:</p>
      <h3>1 · HC18 metadata</h3>
      <p>The HC18 Grand Challenge dataset ships a per-image <code>pixel size(mm)</code> value. When
      the uploaded file's name matches a row in the bundled CSVs, that exact scale is used. Because
      the same filename can appear in both the training and test splits, a collision is resolved by
      comparing the uploaded pixels against the local reference images and choosing the split whose
      image is nearest (mean absolute difference below a small threshold).</p>
      <h3>2 · Manual scale</h3>
      <p>For images outside the HC18 distribution, a manual <code>mm/px</code> value may be entered.
      A "force manual" switch lets the operator deliberately override auto-calibration.</p>
      <h3>3 · Fallback default</h3>
      <p>If neither source is available, a nominal default of <code>{config.DEFAULT_PIXEL_SIZE_MM:.2f} mm/px</code>
      is used and the result is explicitly flagged as <em>not physically calibrated</em>. In this
      state HC is a shape estimate only, and the interface warns accordingly.</p>
      <div class="doc-callout">
        <div class="h">Why it matters</div>
        <p>HC scales linearly with <code>mm_per_px</code>. A 5% calibration error becomes a 5% HC
        error, which can shift the percentile by tens of points near the middle of the distribution.
        Calibration provenance is surfaced on every result for exactly this reason.</p>
      </div>
    </section>

    <section class="doc-sec" id="segmentation">
      <div class="kicker">§ 4</div>
      <h2>Skull segmentation</h2>
      <p>Two exported architectures are offered, both served through <strong>ONNX Runtime</strong>
      (CPU by default, CUDA when available). Each returns a binary skull mask at the original image
      resolution, so the downstream geometry is identical regardless of model. Weights are optional:
      if an <code>.onnx</code> file is absent, the model reports "missing weights" and the rest of
      the interface still functions.</p>
      <h3>YOLOv8-seg — single-stage detector with a prototype-mask head</h3>
      <p>The image is letterboxed to <code>{yolo.input_size[0]}×{yolo.input_size[1]}</code> (aspect
      ratio preserved, padded), normalized by <code>/255</code>, and run through the network. The
      detection head yields boxes, class scores and 32 mask coefficients per anchor; a separate
      branch yields 32 mask prototypes. The highest-confidence detection above a confidence
      threshold of <code>{yolo.conf_threshold}</code> is taken as the skull, its mask is assembled as
      a linear combination of the prototypes (through a sigmoid), restricted to the detection box,
      then un-padded and resized back to the original frame. Fast, single-pass inference.</p>
      <h3>U-Net — encoder–decoder semantic segmentation</h3>
      <p>The image is resized to <code>{unet.input_size[0]}×{unet.input_size[1]}</code> and normalized
      with ImageNet statistics (mean <code>{unet.mean}</code>, std <code>{unet.std}</code>) to match a
      standard training pipeline. The network emits a dense probability map; single-channel and
      two-class output layouts are auto-detected, a sigmoid or softmax is applied only if the export
      produced logits, and the map is thresholded at <code>{pp['mask_threshold']}</code> and resized to the
      original resolution.</p>
      <div class="doc-callout">
        <div class="h">Export contract</div>
        <p>Input size, channel count, and normalization are declared per model in <code>config.py</code>.
        These are the only values that must be adjusted to wire in your own exported weights; the
        inference code adapts to the declared contract rather than hard-coding it.</p>
      </div>
    </section>

    <section class="doc-sec" id="ellipse">
      <div class="kicker">§ 5</div>
      <h2>Contour isolation &amp; ellipse fit</h2>
      <p>The raw mask is binarized and morphologically closed with an elliptical kernel of
      <code>{pp['morph_kernel']}</code> px to seal small gaps. External contours are extracted, and the
      largest is selected — provided it exceeds <code>{pp['min_area_frac']*100:.0f}%</code> of the image
      area and contains at least five points (the minimum for an ellipse fit). If no contour
      qualifies, the image is rejected as "no confident segmentation" rather than producing a
      spurious number.</p>
      <p>An ellipse is then fitted to that contour by direct least squares
      (<code>cv2.fitEllipse</code>). The full major and minor axis lengths and orientation are
      retained so the overlay can redraw the exact fitted ellipse with an axis crosshair for visual
      confirmation of placement.</p>
    </section>

    <section class="doc-sec" id="circumference">
      <div class="kicker">§ 6</div>
      <h2>Head circumference — the Ramanujan II perimeter</h2>
      <p>The ellipse semi-axes are converted to millimetres, <code>a = (major/2)·mm_per_px</code> and
      <code>b = (minor/2)·mm_per_px</code>, and HC is taken as the ellipse perimeter. An ellipse
      perimeter has no elementary closed form, so it is computed with <strong>Ramanujan's second
      approximation</strong>, which is accurate to a few parts per million at realistic fetal-skull
      eccentricities and matches the convention used to derive the HC18 ground truth:</p>
      <div class="doc-formula">
        HC &asymp; &pi;·(a + b)·[ 1 + 3h ⁄ ( 10 + &radic;(4 − 3h) ) ]
        <span class="where">where&nbsp; h = ( (a − b) ⁄ (a + b) )²&nbsp; and&nbsp; a, b are the ellipse semi-axes in mm.</span>
      </div>
      <p>Because the measurement is a perimeter of a fitted ellipse rather than a pixel count around
      a jagged mask, it is stable against small segmentation roughness — the fit averages over local
      boundary noise.</p>
    </section>

    <section class="doc-sec" id="percentile">
      <div class="kicker">§ 7</div>
      <h2>Growth percentile — the Hadlock reference</h2>
      <p>The measured HC is interpreted against a head-circumference-for-gestational-age reference.
      The percentile engine needs only two functions of gestational age (GA): the population
      <strong>mean HC</strong> and its <strong>standard deviation</strong>. The mean follows the
      Hadlock composite biometry chart, tabulated per completed week and linearly interpolated for
      half-weeks; the SD follows a documented linear model,</p>
      <div class="doc-formula">
        &sigma;(GA) &asymp; {ref._SD_SLOPE:.3f}·GA − {abs(ref._SD_INTERCEPT):.3f} mm
        <span class="where">anchored to &asymp; {sd_lo:.0f} mm at {ga_lo} weeks rising to &asymp; {sd_hi:.0f} mm at {ga_hi} weeks, with a {ref._SD_FLOOR_MM:.0f} mm floor.</span>
      </div>
      <p>The reference distribution is treated as Gaussian at each GA. The measurement is converted
      to a standard score and then to a percentile via the normal cumulative distribution function
      Φ (evaluated with <code>math.erf</code>, so there is no SciPy dependency):</p>
      <div class="doc-formula">
        z = (HC − &mu;<sub>GA</sub>) ⁄ &sigma;<sub>GA</sub>&emsp;&emsp; percentile = Φ(z) × 100
      </div>
      <p>The table below is generated directly from the reference module, so it is exactly the curve
      the app evaluates against.</p>
      <div class="doc-table">
        <table>
          <caption>Hadlock HC-for-GA reference (millimetres), {ga_lo}–{ga_hi} completed weeks</caption>
          <thead><tr>
            <th>GA (wk)</th><th>Mean</th><th>SD</th><th>10th</th><th>50th</th><th>90th</th>
          </tr></thead>
          <tbody>{ref_rows_html}</tbody>
        </table>
      </div>
    </section>

    <section class="doc-sec" id="risk">
      <div class="kicker">§ 8</div>
      <h2>Screening risk bands</h2>
      <p>The percentile is mapped to one of three screening bands. Colour is used <em>only</em> to
      encode this clinical meaning — never for decoration.</p>
      <div class="doc-bands">{bands_html}</div>
      <p>These bands are a screening aid, not a diagnosis. Growth restriction is a clinical
      determination that integrates serial measurements, other biometry (abdominal circumference,
      femur length), Dopplers and history — none of which a single HC percentile can replace.</p>
    </section>

    <section class="doc-sec" id="config">
      <div class="kicker">§ 9</div>
      <h2>Reproducibility &amp; configuration</h2>
      <p>Every value that affects a clinical or geometric result is centralized in
      <code>config.py</code> so it can be reviewed in one place: the gestational-age range, the risk
      thresholds ({hi:.0f}th and {md:.0f}th), the segmentation post-processing parameters (mask
      threshold {pp['mask_threshold']}, closing kernel {pp['morph_kernel']} px, minimum area
      {pp['min_area_frac']*100:.0f}%), the per-model export contracts, and the fallback calibration.
      The reference coefficients live in a single, swappable module. Changing a coefficient there
      updates both the app's behaviour and this page — no other code changes are required.</p>
    </section>

    <section class="doc-sec" id="limitations">
      <div class="kicker">§ 10</div>
      <h2>Assumptions &amp; limitations</h2>
      <ul>
        <li><b>Normative reference must be verified.</b> The bundled Hadlock coefficients are a
        faithful, literature-consistent reference for demonstration. For any real use they should be
        independently validated against a cited source (or replaced with an INTERGROWTH-21st table).</li>
        <li><b>Gaussian assumption.</b> Percentiles assume HC is normally distributed at each GA.
        This is reasonable in the normal range but less so far into the tails.</li>
        <li><b>Single measurement.</b> One percentile is a snapshot; growth assessment is inherently
        longitudinal and cannot be replaced by an isolated value.</li>
        <li><b>Segmentation dependence.</b> Output quality tracks image quality and how closely the
        ONNX export matches the configured pre/post-processing. A poor scan or a mismatched export
        yields a poor mask, and therefore a poor measurement.</li>
        <li><b>Calibration dependence.</b> An uncalibrated image (fallback default) gives a shape
        estimate only; the millimetre value is not physically meaningful until a real scale is set.</li>
        <li><b>Not a certified device.</b> Outputs are for engineering demonstration and research
        only and must not be used for clinical decision-making.</li>
      </ul>
    </section>

    <section class="doc-sec" id="references">
      <div class="kicker">References</div>
      <h2>References</h2>
      <ol class="doc-refs">
        <li>Hadlock FP, Deter RL, Harrist RB, Park SK. <i>Estimating fetal age: computer-assisted
        analysis of multiple fetal growth parameters.</i> Radiology 1984;152:497–501.</li>
        <li>van den Heuvel TLA, de Bruijn D, de Korte CL, van Ginneken B. <i>Automated measurement of
        fetal head circumference using 2D ultrasound images (HC18 Grand Challenge).</i> PLoS ONE 2018.</li>
        <li>Ramanujan S. <i>Modular equations and approximations to &pi;.</i> Quarterly Journal of
        Mathematics 1914 — second perimeter approximation for the ellipse.</li>
        <li>Chitty LS, Altman DG, Henderson A, Campbell S. <i>Charts of fetal size: 2. Head
        measurements.</i> BJOG 1994;101:35–43 — normal-range cross-check.</li>
        <li>Jocher G, et&nbsp;al. <i>Ultralytics YOLOv8.</i> 2023 — segmentation architecture.</li>
        <li>Ronneberger O, Fischer P, Brox T. <i>U-Net: Convolutional Networks for Biomedical Image
        Segmentation.</i> MICCAI 2015.</li>
      </ol>
    </section>

    <div class="doc-footer">
      {config.APP_NAME} · v{config.APP_VERSION} — {config.APP_TAGLINE}<br>
      This methodology is generated from the running clinical modules; displayed coefficients and
      thresholds reflect the code in this build.
    </div>

  </div>
</div>
"""

st.markdown(DOC, unsafe_allow_html=True)
