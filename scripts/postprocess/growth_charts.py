"""Hadlock Fetal Growth Reference Table and Percentile Interpolation."""
import numpy as np

# Hadlock (1984) reference data: week -> [3rd, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 97th] percentile values (mm)
HADLOCK_CHART = {
    14: [85, 88, 93, 101, 110, 119, 127, 132, 135],
    16: [109, 113, 118, 127, 137, 147, 156, 161, 165],
    18: [133, 137, 143, 153, 164, 175, 185, 191, 195],
    20: [156, 161, 168, 179, 191, 203, 214, 221, 226],
    22: [179, 184, 192, 204, 218, 232, 244, 252, 257],
    24: [201, 207, 215, 229, 244, 259, 273, 281, 287],
    26: [222, 229, 238, 253, 269, 285, 300, 309, 316],
    28: [243, 250, 259, 276, 293, 310, 327, 337, 344],
    30: [262, 270, 280, 298, 316, 334, 352, 362, 370],
    32: [280, 289, 299, 318, 338, 358, 376, 387, 395],
    34: [297, 306, 317, 337, 358, 379, 399, 410, 419],
    36: [312, 321, 333, 354, 376, 398, 419, 431, 440],
    38: [325, 335, 347, 369, 392, 415, 437, 449, 459],
    40: [336, 346, 359, 382, 406, 430, 453, 466, 476],
}

PERCENTILE_MARKERS = [3.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 97.0]


def interpolate_growth_thresholds(gestational_age_weeks: float) -> np.ndarray:
    """Linearly interpolate Hadlock HC thresholds for fractional gestational ages."""
    sorted_weeks = sorted(HADLOCK_CHART.keys())
    
    if gestational_age_weeks < sorted_weeks[0] or gestational_age_weeks > sorted_weeks[-1]:
        raise ValueError(
            f"Gestational age {gestational_age_weeks} weeks is outside "
            f"supported Hadlock range ({sorted_weeks[0]} - {sorted_weeks[-1]} weeks)."
        )

    # Exact week match
    if gestational_age_weeks in HADLOCK_CHART:
        return np.array(HADLOCK_CHART[gestational_age_weeks], dtype=float)

    # Find lower and upper bound weeks
    idx = np.searchsorted(sorted_weeks, gestational_age_weeks)
    w_low = sorted_weeks[idx - 1]
    w_high = sorted_weeks[idx]

    # Compute interpolation weight
    t = (gestational_age_weeks - w_low) / (w_high - w_low)
    
    thresholds_low = np.array(HADLOCK_CHART[w_low], dtype=float)
    thresholds_high = np.array(HADLOCK_CHART[w_high], dtype=float)

    return (1.0 - t) * thresholds_low + t * thresholds_high


def calculate_growth_percentile(hc_mm: float, gestational_age_weeks: float) -> float:
    """
    Determine the exact fetal head growth percentile using linear interpolation.

    Args:
        hc_mm: Fetal head circumference in mm.
        gestational_age_weeks: Gestational age in weeks.

    Returns:
        Estimated percentile score between 0.0 and 100.0.
    """
    thresholds = interpolate_growth_thresholds(gestational_age_weeks)

    # 1. Edge Case: Below the 3rd percentile
    if hc_mm <= thresholds[0]:
        # Extrapolate downwards down to 0 percentile
        t = hc_mm / thresholds[0]
        return max(0.0, t * 3.0)

    # 2. Edge Case: Above the 97th percentile
    if hc_mm >= thresholds[-1]:
        # Extrapolate upwards capping at 100
        diff = hc_mm - thresholds[-1]
        percentile = 97.0 + (3.0 * (1.0 - np.exp(-diff / 50.0)))  # asymptotic growth
        return min(100.0, percentile)

    # 3. Intermediate interpolation: find which percentile bins the HC falls between
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= hc_mm < thresholds[i + 1]:
            val_low, val_high = thresholds[i], thresholds[i + 1]
            p_low, p_high = PERCENTILE_MARKERS[i], PERCENTILE_MARKERS[i + 1]
            # Interpolate
            t = (hc_mm - val_low) / (val_high - val_low)
            percentile = p_low + t * (p_high - p_low)
            return round(percentile, 2)
            
    return 50.0 # fallback


def classify_clinical_risk(percentile: float) -> tuple[str, str]:
    """
    Classify growth risk category based on percentile ranking.

    Returns:
        Tuple: (Risk Category String, Hex Color Code)
    """
    if percentile < 10.0:
        return "HIGH RISK (IUGR Screening Flagged)"
    elif percentile < 25.0:
        return "MEDIUM RISK (Monitor Closely)"
    else:
        return "NORMAL GROWTH"
    
def main():
    thresholds = interpolate_growth_thresholds(29.4)
    print(f"Thresholds: {thresholds}")
    percentile = calculate_growth_percentile(283, 29.4)
    print(f"Percentile: {percentile}")
    risk = classify_clinical_risk(percentile)
    print(f"Risk: {risk}")


if __name__ == "__main__":
    main()
