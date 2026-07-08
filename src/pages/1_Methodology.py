"""
pages/1_Methodology.py
======================
The methodology as its own page (Streamlit multipage format). It is reachable
from the sidebar navigation and from the prominent "Methodology" link in the
homepage masthead/empty state.

The document is written in a medical-journal register and — importantly — pulls
its parameter figures directly from the live clinical modules (``reference_hadlock``,
``config``, ``percentiles``). The Hadlock reference chart, the standard deviation
model, the risk thresholds, and the model export specifications shown below
therefore always match the code that actually runs; editing a configuration
value in one place updates the methodology page automatically.
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

import streamlit as st  # noqa: E402  (must be after set_page_config)


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

# Hadlock reference chart (integer completed weeks) — generated dynamically from the code.
_ref_rows = []
for wk in range(ga_lo, ga_hi + 1):
    s = ref.reference_summary(wk)
    _ref_rows.append(
        f"<tr><td>{wk}</td><td>{s.mean_mm:.0f}</td><td>{s.sd_mm:.1f}</td>"
        f"<td>{s.p10_mm:.0f}</td><td>{s.p50_mm:.0f}</td><td>{s.p90_mm:.0f}</td></tr>"
    )
ref_rows_html = "".join(_ref_rows)

# Risk-band rows, coloured by the same semantic tokens the application uses.
bands = [
    (rc["HIGH"]["solid"], f"&lt; {hi:.0f}th", "High Risk — Screening Alert Threshold",
     "Identifies growth profiles residing below the 10th percentile, which warrants screening for "
     "intrauterine growth restriction (IUGR). Clinical correlation and supplementary diagnostic evaluations "
     "(e.g., Doppler velocimetry, amniotic fluid volume assessment) are recommended. An isolated screening outcome "
     "warrants clinical review rather than immediate definitive diagnostic classification."),
    (rc["MEDIUM"]["solid"], f"{hi:.0f}–{md:.0f}th", "Medium Risk — Borderline Growth Range",
     "Identifies growth profiles residing in the lower-normal boundary of the normative distribution. Serial growth "
     "assessments via diagnostic ultrasonography are recommended to differentiate normal physiological variants from early "
     "manifestations of pathological growth deceleration."),
    (rc["NORMAL"]["solid"], f"&ge; {md:.0f}th", "Normal Growth Profile",
     "Indicates biometric dimensions residing within the anticipated growth distribution curve for the designated gestational age."),
]
bands_html = "".join(
    f'<div class="doc-band" style="--rk:{c};">'
    f'<div class="tag">{tag}</div>'
    f'<div class="bd"><b>{title}</b><p>{body}</p></div></div>'
    for c, tag, title, body in bands
)


# --------------------------------------------------------------------------- #
# Document Output
# --------------------------------------------------------------------------- #
DOC = f"""
<div class="doc">

  <a class="back-link" href="/" target="_self">{C.icon('arrow_left')}<span>Back to analyzer</span></a>

  <div class="doc-hero">
    <div class="eyebrow">Methodological Framework &amp; Clinical Reference</div>
    <h1>Methodological Framework for Automated Fetal Head Circumference Estimation and Growth Restriction Screening</h1>
    <p class="lede">This document presents a formal, end-to-end technical account of the algorithmic and statistical pipeline designed to automate fetal head circumference (HC) estimation from two-dimensional ultrasound scans. The methodology describes spatial scale calibration, deep semantic skull segmentation, morphological boundary isolation, least-squares ellipse fitting, parametric perimeter estimation, and clinical growth percentile mapping.</p>
    <div class="doc-meta">
      <span><b>Reference Model</b> Hadlock HC-for-GA (1984)</span>
      <span><b>Gestational Age Limits</b> {ga_lo}–{ga_hi} completed weeks</span>
      <span><b>Classification</b> Research Prototype</span>
    </div>
  </div>

  <div class="doc-toc">
    <div class="t">Contents</div>
    <ol>
      <li><a href="#overview">Introduction &amp; Clinical Context</a></li>
      <li><a href="#pipeline">Computational Pipeline Overview</a></li>
      <li><a href="#data-pipeline">Data Corpus &amp; Preprocessing Pipeline</a></li>
      <li><a href="#segmentation">Deep Learning Segmentation Architectures</a></li>
      <li><a href="#calibration">Spatial Calibration &amp; Scale Resolution</a></li>
      <li><a href="#ellipse">Morphological Filtering &amp; Ellipse Fitting</a></li>
      <li><a href="#circumference">Perimeter Estimation via Ramanujan II Approximation</a></li>
      <li><a href="#percentile">Growth Percentile Evaluation &amp; Statistical Distribution</a></li>
      <li><a href="#risk">Clinical Risk Classification &amp; Screening Bands</a></li>
      <li><a href="#config">Centralized Configuration &amp; Reproducibility</a></li>
      <li><a href="#verification">System Validation, Testing &amp; Limitations</a></li>
    </ol>
  </div>

  <div class="doc-body">

    <section class="doc-sec" id="overview">
      <div class="kicker">Section 1</div>
      <h2>Introduction &amp; Clinical Context</h2>
      <p>Intrauterine growth restriction (IUGR) is a significant pathology in obstetrics, predisposing neonates to elevated risks of perinatal morbidity, neurodevelopmental delays, and mortality. Accurate screening for IUGR depends heavily on the evaluation of fetal growth parameters, among which <strong>fetal head circumference (HC)</strong> is a critical metric. The measurement of HC at the trans-thalamic plane provides essential diagnostic data regarding cranial development and gestational age matching.</p>
      <p>In current clinical practice, head circumference is manually estimated by sonographers using interactive ellipse overlays on ultrasound systems. This manual workflow exhibits three primary limitations: it is highly operator-dependent, introduces inter-observer variability ranging from 5% to 10%, and demands substantial time from skilled personnel, which limits its throughput in low-resource and rural clinics. To mitigate these issues, the {config.APP_NAME}-AI framework implements an automated, objective, and auditable methodology for cranial segmentation, geometric reconstruction, and clinical percentile calculation.</p>
      <div class="doc-callout" style="--rk:{rc['MEDIUM']['solid']};">
        <div class="h">{C.icon('warning')} Regulatory Notice — Research Prototype</div>
        <p>{config.SAFETY_NOTICE}</p>
      </div>
    </section>

    <section class="doc-sec" id="pipeline">
      <div class="kicker">Section 2</div>
      <h2>Computational Pipeline Overview</h2>
      <p>The system processes an input two-dimensional ultrasound image of the fetal head through five sequential, deterministic phases. The pipeline is designed for auditability, where each component executes a discrete coordinate or intensity transform that can be validated independently:</p>
      <ol class="doc-steps">
        <li><b>Spatial Calibration.</b> Resolution of the physical scaling factor (expressed in millimetres per pixel) to establish a mapping from coordinate space to physical space.</li>
        <li><b>Semantic Segmentation.</b> Inference via a deep convolutional neural network (YOLOv8-seg or U-Net) to generate a dense, binary probability mask of the fetal skull.</li>
        <li><b>Geometric Post-processing.</b> Application of morphological filters to seal segmentation voids, followed by Suzuki-Abe contour extraction to isolate the primary anatomical boundary.</li>
        <li><b>Parametric Reconstruction.</b> Fitting of a parametric ellipse model to the isolated boundary coordinates via a direct least-squares estimator.</li>
        <li><b>Clinical Interpretation.</b> Computation of the ellipse perimeter using Ramanujan's second approximation, conversion to physical millimetres, and evaluation against gestational growth curves.</li>
      </ol>
    </section>

    <section class="doc-sec" id="data-pipeline">
      <div class="kicker">Section 3</div>
      <h2>Data Corpus &amp; Preprocessing Pipeline</h2>
      <p>The segmentation models were trained, validated, and evaluated using the public <strong>HC18 Grand Challenge dataset</strong>, which contains 999 training images and 335 test images (with labels withheld for evaluation) representing variable-resolution ultrasound scans of the fetal head. Annotations are provided as ground-truth ellipse parameters associated with physical spatial calibration metrics. The dataset was split into training and validation sets using an 85/15 ratio (random seed = 67), yielding 849 training pairs and 150 validation pairs. The validation set was used for hyperparameter tuning and early stopping optimization.</p>
      <p>The preprocessing pipeline executes the following sequence:
      <ol>
        <li><b>Ellipse to Mask Render:</b> Ground-truth ellipse coordinates are parsed and rendered into binary target masks of identical dimensions to the raw scans.</li>
        <li><b>Vector Polygon Conversion:</b> For YOLOv8-seg training, target masks are converted into coordinate sequences mapping the skull boundary.</li>
        <li><b>Data Augmentation:</b> To improve generalization across varying scan environments, five stochastic augmentations are applied during training:
          <ul>
            <li><i>Horizontal Flip:</i> Applied with a probability of 50% to achieve invariance to fetal orientation.</li>
            <li><i>Random Rotation:</i> Applied within a ±15-degree range to simulate transducer angle variations.</li>
            <li><i>Scaling Factor:</i> Adjusted between 0.8x and 1.2x to replicate biological size variations.</li>
            <li><i>Brightness Adjustment:</i> Varied within a ±30% range to simulate differing acoustic power and system gain.</li>
            <li><i>Contrast Adjustment:</i> Perturbed within a ±20% range to mimic signal attenuation differences.</li>
          </ul>
        </li>
      </ol>
      </p>
    </section>

    <section class="doc-sec" id="segmentation">
      <div class="kicker">Section 4</div>
      <h2>Deep Learning Segmentation Architectures</h2>
      <p>The system provides two distinct model options for skull boundary segmentation. Both architectures are compiled into the <strong>Open Neural Network Exchange (ONNX)</strong> format and executed via <strong>ONNX Runtime</strong> for efficient, hardware-agnostic inference (&lt;200ms target execution):</p>
      <h3>1 · YOLOv8s-seg (Primary Architecture)</h3>
      <p>The primary model is a single-stage instance-segmentation network containing 11.8 million parameters. It employs a CSPDarknet53 backbone for multi-scale feature extraction, a Path Aggregation Network (PANet) neck to fuse features across scales, and decoupled detection and segmentation heads. The segmentation head generates 32 prototype masks at one-fourth of the input resolution. The detection head predicts class scores, bounding boxes, and 32 mask coefficients per anchor. The highest-confidence detection above the threshold of <code>{yolo.conf_threshold}</code> is selected, and its mask is computed as a linear combination of the prototype masks and coefficients. The mask is then thresholded, restricted to the bounding box, and upscaled to the original image dimensions. The network expects input images resized to <code>{yolo.input_size[0]}×{yolo.input_size[1]}</code> pixels, normalized to <code>[0.0, 1.0]</code>.</p>
      <h3>2 · U-Net ResNet34 (Baseline Architecture)</h3>
      <p>The baseline model is a semantic segmentation network containing 24.4 million parameters. It utilizes a ResNet34 encoder pretrained on ImageNet, coupled to a symmetric decoder with skip connections that transfer high-resolution spatial features from the encoder to the decoder. The network receives inputs resized to <code>{unet.input_size[0]}×{unet.input_size[1]}</code> pixels, normalized using ImageNet channel mean (<code>{unet.mean}</code>) and standard deviation (<code>{unet.std}</code>) statistics. The network outputs a single-channel probability map, which is thresholded at <code>{pp['mask_threshold']}</code> to isolate the skull mask before being resized to the original image dimensions.</p>
    </section>

    <section class="doc-sec" id="calibration">
      <div class="kicker">Section 5</div>
      <h2>Spatial Calibration &amp; Scale Resolution</h2>
      <p>Physical metrics cannot be derived without mapping the digital pixel grid to physical dimensions. The system determines the physical pixel size (expressed in millimetres per pixel) using a hierarchical, deterministic resolution protocol:
      <h3>1 · Database Metadata Matching</h3>
      <p>For scans matching the HC18 dataset, the system queries the physical pixel size from the challenge metadata. If filename collisions occur between splits, a mean absolute difference (MAD) check is performed between the uploaded pixel intensity matrix and local reference images; the calibration corresponding to the image with the minimal MAD is selected.</p>
      <h3>2 · Operator Overrides</h3>
      <p>The system allows clinical operators to manually input known calibration values via the user interface. Activating the manual override switch bypasses automated metadata lookup.</p>
      <h3>3 · Fallback Calibration</h3>
      <p>If neither source is available, the system falls back to a default calibration value of <code>{config.DEFAULT_PIXEL_SIZE_MM:.2f} mm/pixel</code>. Under fallback conditions, the system displays a prominent visual warning indicating that subsequent metrics are uncalibrated and serve as relative morphological shape estimates rather than physical clinical indices.</p>
    </section>

    <section class="doc-sec" id="ellipse">
      <div class="kicker">Section 6</div>
      <h2>Morphological Filtering &amp; Ellipse Fitting</h2>
      <p>The raw binary mask from the segmentation stage is binarized and subjected to morphological post-processing. A morphological closing operation using an elliptical structuring element of kernel size <code>{pp['morph_kernel']}</code> pixels is executed to eliminate intra-boundary voids and reconnect minor skull segmentation discontinuities.
      Following morphological refinement, external contours are extracted using the topological contour retrieval algorithm proposed by Suzuki and Abe. The largest external contour is selected as the representative candidate for the fetal skull. To prevent fitting ellipses to noise or artifacts, candidate contours must satisfy two strict criteria:
      <ul>
        <li><b>Area Constraint:</b> The contour area must exceed a minimum fraction (<code>{pp['min_area_frac']*100:.0f}%</code>) of the total image canvas area.</li>
        <li><b>Coordinate Count:</b> The contour must consist of at least five independent coordinate points, which is the mathematical minimum required for a multi-variate quadric fit.</li>
      </ul>
      Once validated, the parameter space of the ellipse is computed using a direct least-squares estimator (<code>cv2.fitEllipse</code>). This fitting minimizes the algebraic distance between the contour coordinates and the conic equation. The estimator outputs the centroid coordinates $(x_c, y_c)$, the length of the major and minor axes in pixels, and the rotation angle $\theta$. The major axis length is constrained to be greater than or equal to the minor axis length.</p>
    </section>

    <section class="doc-sec" id="circumference">
      <div class="kicker">Section 7</div>
      <h2>Perimeter Estimation via Ramanujan II Approximation</h2>
      <p>The ellipse semi-axes in physical units are computed by applying the calibration factor:</p>
      <div class="doc-formula">
        <i>a</i> = (major_axis_px ⁄ 2) &middot; mm_per_px<br>
        <i>b</i> = (minor_axis_px ⁄ 2) &middot; mm_per_px
      </div>
      <p>Because the perimeter of an ellipse has no closed-form representation in elementary functions, the system employs Ramanujan's second approximation. This approximation is highly efficient and accurate to within a few parts per million (ppm) for the eccentricity range typical of the human fetal skull:
      </p>
      <div class="doc-formula">
        HC &asymp; &pi;·(a + b)·[ 1 + 3h ⁄ ( 10 + &radic;(4 &minus; 3h) ) ]
        <span class="where">where&nbsp; <i>h</i> = ( (<i>a</i> &minus; <i>b</i>) ⁄ (<i>a</i> + <i>b</i>) )&sup2;&nbsp; and&nbsp; <i>a</i>, <i>b</i> are the ellipse semi-axes in mm.</span>
      </div>
      <p>Approximating the perimeter via a parametric ellipse fit rather than computing a direct contour perimeter offers a significant clinical advantage: it acts as a regularizing prior, smoothing out local segmentation noise, pixelation steps, and boundary jitter.</p>
    </section>

    <section class="doc-sec" id="percentile">
      <div class="kicker">Section 8</div>
      <h2>Growth Percentile Evaluation &amp; Statistical Distribution</h2>
      <p>The computed head circumference is mapped to a growth percentile using a normative population distribution. The system implements the Hadlock (1984) composite fetal-biometry growth model, which characterizes the head circumference distribution at a given gestational age (GA) as a Gaussian profile:</p>
      <div class="doc-formula">
        HC &sim; <i>N</i>(&mu;<sub>GA</sub>, &sigma;<sub>GA</sub>&sup2;)
      </div>
      <p>The population mean, &mu;<sub>GA</sub>, is calculated by linear interpolation between the weekly clinical mean values tabulated in the Hadlock reference. The standard deviation, &sigma;<sub>GA</sub>, is modeled as a linear function of gestational age:
      </p>
      <div class="doc-formula">
        &sigma;(GA) &asymp; {ref._SD_SLOPE:.4f}·GA - {abs(ref._SD_INTERCEPT):.4f} mm
        <span class="where">anchored to &asymp; {sd_lo:.1f} mm at {ga_lo} weeks rising to &asymp; {sd_hi:.1f} mm at {ga_hi} weeks, with a {ref._SD_FLOOR_MM:.1f} mm floor.</span>
      </div>
      <p>To convert the physical measurement into a standard score (z-score) and subsequently into a percentile, the cumulative distribution function (CDF) of the standard normal distribution &Phi;(z) is evaluated:</p>
      <div class="doc-formula">
        z = (HC - &mu;<sub>GA</sub>) ⁄ &sigma;<sub>GA</sub>&emsp;&emsp; percentile = &Phi;(z) × 100
        <span class="where">where &Phi;(z) is the standard normal cumulative distribution function evaluated using the error function (erf).</span>
      </div>
      <p>The error function (erf) is computed via <code>math.erf</code> to prevent external library dependencies. Calculated percentiles are clamped to the range [0.1, 99.9] for display purposes. The table below displays the population values generated dynamically by the reference module:</p>
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
      <div class="kicker">Section 9</div>
      <h2>Clinical Risk Classification &amp; Screening Bands</h2>
      <p>To support clinical workflows, the growth percentile is categorized into three clinical screening risk bands based on established clinical guidance:</p>
      <div class="doc-bands">{bands_html}</div>
      <p>These risk designations are designed as screening aids and should not be used as standalone diagnostic results. Definitive clinical assessment requires the integration of longitudinal measurements, multi-biometric parameters (abdominal circumference, femur length, biparietal diameter), uterine/umbilical artery Dopplers, and clinical history.</p>
    </section>

    <section class="doc-sec" id="config">
      <div class="kicker">Section 10</div>
      <h2>Centralized Configuration &amp; Reproducibility</h2>
      <p>To ensure clinical auditability, reproducibility, and system configurability, all parameters affecting the segmentation, calibration, post-processing, and risk classification stages are isolated in a centralized module (<code>config.py</code>). This decoupling allows clinical researchers and engineers to adjust key thresholds—such as segmentation confidence levels, morphological kernel sizes, minimum contour area fractions, and risk thresholds—in a single file, immediately propagating these updates throughout both the runtime inference logic and the generated methodology documentation.</p>
    </section>

    <section class="doc-sec" id="verification">
      <div class="kicker">Section 11</div>
      <h2>System Validation, Testing &amp; Limitations</h2>
      <p>The system undergoes automated validation using test scripts located in the <code>scripts/test/</code> directory. These scripts run inference on a validation split (<code>data/preprocessed/fastai/images/val/</code>) and compare predicted segmentations and fitted ellipses against ground-truth parameters using standard metrics, including:
      <ul>
        <li><b>Dice Similarity Coefficient (DSC):</b> Evaluates overlap accuracy between predicted and ground-truth masks.</li>
        <li><b>Mean Absolute Error (MAE) in HC:</b> Quantifies the average absolute difference in millimetres between estimated and ground-truth head circumferences.</li>
      </ul>
      The framework's operation is subject to several technical and clinical limitations:
      <ol>
        <li><b>Normative Curve Calibration:</b> The Hadlock references are configured for demonstration and must be validated against the specific demographic patient population under care.</li>
        <li><b>Gaussian Tail Sensitivity:</b> The Gaussian model is a standard approximation but may exhibit limitations at extreme tail distributions (&lt; 1st or &gt; 99th percentiles).</li>
        <li><b>Sensitivity to Acoustic Artifacts:</b> Shadowing, bone attenuation, and low contrast in poor-quality ultrasound scans can degrade segmentation accuracy, directly propagating error into the ellipse fitting stage.</li>
        <li><b>Resolution and Calibration Dependency:</b> Accurate physical scale representation is critically dependent on physical calibration metadata; uncalibrated scans only permit relative shape assessments.</li>
        <li><b>Cross-Sectional Limitation:</b> A single-point measurement is a static snapshot and cannot replace longitudinal tracking of fetal growth over time.</li>
      </ol>
      </p>
      
      <div class="kicker">References</div>
      <h3>References</h3>
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
      {config.APP_NAME} — {config.APP_TAGLINE}<br>
    </div>
  </div>
</div>
"""

st.html(DOC)
