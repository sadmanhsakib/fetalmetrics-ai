# Methodology — FetalMetrics AI

This document records the research design decisions behind every stage of the FetalMetrics AI pipeline — why each approach was chosen, what the alternatives were, and what the known limitations are. It is intended for technically literate readers, clinical researchers, and data scientists who wish to understand the algorithmic and statistical reasoning behind the system, rather than just its final diagnostic outputs.

---

## 1. Problem Framing

### 1.1 Why This Is a Hard Problem

Automated fetal biometry estimation from two-dimensional (2D) trans-thalamic ultrasound images presents a significant computational challenge. Fetal head circumference (HC) is a primary metric for assessing gestational age and screening for intrauterine growth restriction (IUGR). However, automating its measurement is constrained by several structural factors:

1. **Acoustic Artifacts & Image Quality**: Ultrasound is inherently noisy. Scans exhibit acoustic shadowing, acoustic enhancement, speckle noise, and signal attenuation from fetal bone structures. 
2. **Anatomical Ambiguity**: The boundaries of the fetal skull are often incomplete due to the angle of insonation, presenting discontinuous edges that fool traditional gradient-based edge detectors.
3. **Clinical Tolerance**: Clinical applications require sub-millimeter precision. A deviation of a few pixels in the semi-axes can shift a growth percentile dramatically, altering the risk assessment.
4. **Scale Calibration**: A raw pixel grid has no intrinsic physical scale. Precise mapping to millimeters requires robust metadata parsing or physical calibration fallback.

Manual measurement by sonographers takes time, suffers from inter-observer variability (typically 5–10%), and relies on subjective operator judgment. 

### 1.2 Why a Deep Learning Approach?

Given the challenges of ultrasound imaging, three computational paradigms were evaluated:

1. **Traditional Computer Vision**: Approaches such as Canny edge detection followed by the Hough Transform. 
   *Rejected*: Highly sensitive to speckle noise and acoustic shadowing. Discontinuous skull boundaries cause severe under-fitting or catastrophic failures in Hough space.
2. **Active Contour Models (Snakes)**: Energy-minimizing splines guided by external image forces.
   *Rejected*: Requires a manual, well-placed initialization seed close to the true boundary. Fails to converge autonomously on highly attenuated skull boundaries.
3. **Deep Convolutional Neural Networks (CNNs)**: 
   *Selected*: Deep learning approaches bypass local gradient reliance by learning the global semantic context of a fetal skull. CNNs are inherently robust to speckle noise and can implicitly learn to "hallucinate" or bridge missing boundary segments where acoustic shadowing occurs.

---

## 2. Data Corpus & Preprocessing

### 2.1 Dataset Selection — HC18 Grand Challenge

The HC18 Grand Challenge dataset was selected as the foundational training corpus. This dataset satisfies four critical constraints:
- **Ecological Validity**: Contains real-world ultrasound scans collected under varying clinical conditions, rather than synthetic or idealized data.
- **Robust Annotations**: Includes expert-annotated ground-truth ellipse parameters (center coordinates, semi-axes, angle).
- **Physical Calibration**: Includes pixel-to-millimeter scaling metadata for every scan, enabling true physical measurement learning.
- **Volume**: Comprises 999 training images and 335 test images, providing sufficient statistical power for deep learning convergence.

### 2.2 Dataset Splitting Strategy

The dataset was partitioned into a training set (849 images) and a validation set (150 images) using an 85/15 ratio, anchored by a fixed random seed (`67`). 
Random cross-validation was deemed unnecessary given the large corpus size; a fixed static split guarantees reproducible hyperparameter tuning and model comparison across different architectural experiments.

### 2.3 Preprocessing & Augmentation Pipeline

Deep networks are highly prone to overfitting on medical imaging due to homogeneous textures. To force the network to learn invariant anatomical features rather than noise patterns, we applied the following stochastic augmentations during training:

- **Horizontal Flip (50% probability)**: Ensures invariance to fetal head orientation (left vs. right facing). Note: Vertical flips were omitted to preserve the anatomically standard "up/down" acoustic reflection characteristics.
- **Random Rotation (±15°)**: Simulates transducer angle variations typical in clinical practice.
- **Scaling (0.8× to 1.2×)**: Mimics natural variations in gestational age and biological size.
- **Brightness (±30%) & Contrast (±20%)**: Replicates variations in acoustic power, system gain settings, and machine-specific transducer impedance.

Ground-truth ellipses were converted to dense binary target masks to facilitate semantic and instance segmentation architectures.

---

## 3. Deep Learning Architectures

The system integrates two distinct architectures, both compiled to the Open Neural Network Exchange (ONNX) format for deterministic, hardware-agnostic execution via ONNX Runtime.

### 3.1 YOLOv8s-seg (Primary Architecture)

The primary model is a single-stage instance-segmentation network (11.8 million parameters). 

- **Architecture**: Employs a CSPDarknet53 backbone for multi-scale feature extraction, leading into a Path Aggregation Network (PANet) neck. It utilizes decoupled heads for detection (bounding box) and segmentation (mask coefficients).
- **Mask Generation**: The segmentation head generates 32 prototype masks at 1/4 input resolution. The final mask is reconstructed as a linear combination of these prototypes weighted by predicted coefficients.
- **Rationale**: YOLOv8 was selected over Mask R-CNN because of its speed-to-accuracy Pareto efficiency. It achieves state-of-the-art segmentation with sub-100ms inference times on CPU, fulfilling the requirement for real-time edge deployment.

### 3.2 U-Net ResNet34 (Baseline Architecture)

The secondary model is a classic semantic segmentation network (24.4 million parameters).

- **Architecture**: A ResNet34 encoder pretrained on ImageNet is coupled to a symmetric decoder with spatial skip connections.
- **Rationale**: U-Net serves as the gold-standard baseline in medical image segmentation. While slightly heavier and slower than YOLOv8s-seg, it provides a highly stable dense pixel probability map, useful as a fallback or comparative benchmark.

---

## 4. Geometric Post-Processing Pipeline

Extracting a clinical measurement from a probability map requires transitioning from discrete pixel space to a continuous parametric model.

### 4.1 Morphological Filtering & Contour Extraction

The raw binary mask invariably contains minor discontinuities. 
1. **Morphological Closing**: We apply a closing operation using an elliptical structuring element (kernel size = 5px). This eliminates intra-boundary voids and smooths pixelated edges.
2. **Topological Retrieval**: The Suzuki-Abe contour extraction algorithm isolates the largest external boundary. 
3. **Artifact Rejection**: To prevent fitting an ellipse to acoustic noise, the target contour must exceed 10% of the image canvas area and contain ≥ 5 discrete coordinates (the mathematical minimum for a unique quadric fit).

### 4.2 Least-Squares Ellipse Fitting

We compute the parameter space of the fetal skull using a direct least-squares estimator (`cv2.fitEllipse`). This algebraic fit minimizes the orthogonal distance between the discrete contour coordinates and the continuous conic equation. This step acts as a powerful geometric regularizer, enforcing an anatomical prior (an ellipse) onto the noisy segmentation mask.

### 4.3 Perimeter Estimation via Ramanujan II

The perimeter of an ellipse cannot be expressed in closed-form elementary functions. While numerical integration (complete elliptic integral of the second kind) is exact, it is computationally heavy.
We utilize **Ramanujan's second approximation**:

`C ≈ π(a + b) [ 1 + 3h / (10 + √(4 - 3h)) ]` where `h = (a - b)² / (a + b)²`

This approximation is mathematically elegant, deterministic, and accurate to within a few parts per million (ppm) for the mild eccentricities of the human fetal skull, matching the HC18 grand challenge ground-truth conventions precisely.

---

## 5. Clinical Integration & Statistical Mapping

### 5.1 Normative Distribution Modeling (Hadlock 1984)

An absolute measurement in millimeters has limited clinical utility without age-matched context. The system maps the head circumference to a growth percentile using the Hadlock (1984) composite fetal-biometry model.

The distribution of HC at a given gestational age (GA) is modeled as a Gaussian:
`HC ~ N(μ_GA, σ²_GA)`

- **Mean (μ_GA)**: Derived via linear interpolation between the established weekly clinical means.
- **Variance (σ_GA)**: Modeled as a linear function of GA to reflect increasing natural biological variation in later trimesters.

The percentile is computed strictly by evaluating the Cumulative Distribution Function (CDF) of the standard normal distribution using the error function (`math.erf`), isolating the logic from external statistical library dependencies.

### 5.2 Clinical Risk Stratification

To support triage workflows, percentiles are binned into categorical screening bands:
- **High Risk (< 10th percentile)**: Triggers an IUGR screening alert. Warrants clinical correlation and supplementary diagnostics (e.g., Doppler velocimetry).
- **Medium Risk (10th–25th percentile)**: Borderline growth. Suggests serial ultrasonographic assessments to monitor trajectory.
- **Normal (≥ 25th percentile)**: Nominal biometry within expected biological variance.

---

## 6. System Validation Strategy

The pipeline is validated deterministically via isolated test scripts.

### 6.1 Evaluation Metrics

- **Dice Similarity Coefficient (DSC)**: Measures the spatial overlap accuracy between the predicted segmentation mask and the human-annotated ground-truth mask. Represents pixel-level accuracy.
- **Mean Absolute Error (MAE)**: Quantifies the absolute physical deviation (in millimeters) between the algorithmically fitted perimeter and the clinical ground-truth. Represents clinical utility.

### 6.2 Test Harness

The `scripts/test/` directory contains automated validation routines that feed the validation split through the full ONNX inference and post-processing pipeline. This guarantees that any configuration change in `config.py` is immediately auditable against the core metrics.

---

## 7. Known Limitations

This system is a **research prototype** and must be evaluated with full awareness of its methodological constraints.

### 7.1 Technical Limitations

1. **Calibration Dependency**: Physical measurements are mathematically reliant on the `mm_per_px` metadata. If the provided calibration is erroneous, the absolute millimeter output will be proportionally skewed. If no calibration is available, the system falls back to a default relative scale, rendering the output useless for clinical percentile mapping.
2. **Gaussian Tail Breakdown**: The standard normal approximation for population distributions works beautifully between the 5th and 95th percentiles. However, biological traits often exhibit fat tails. Percentiles at the extreme margins (<1st or >99th) carry higher statistical uncertainty and should not be over-interpreted.
3. **Post-Processing Rigidity**: The ellipse fitting pipeline assumes the largest contour is the skull. In scans with severe artifacts or shadowing where a limb bone appears larger and more continuous than the skull fragment, the system will incorrectly fit the limb.

### 7.2 Clinical Limitations

1. **Non-Diagnostic**: This software is not regulatory-approved (FDA/CE/TGA) for clinical diagnostic use. It is a prioritization and research tool.
2. **Cross-Sectional vs. Longitudinal**: The tool processes a single static snapshot. True IUGR diagnosis requires longitudinal tracking of growth velocity across multiple scans, alongside complementary biometry (abdominal circumference, femur length).
3. **Population Calibration**: The embedded Hadlock model represents a specific demographic historical cohort. To be clinically actionable, normative curves must be calibrated to the specific demographic population being served by the clinic.

---

*This document serves as the academic foundation for the application. For deployment instructions, refer to the README. For inline module logic, refer to the docstrings within the source directory.*