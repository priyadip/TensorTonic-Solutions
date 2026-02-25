def detect_drift(reference_counts, production_counts, threshold):
    # Normalize histograms
    ref_total = sum(reference_counts)
    prod_total = sum(production_counts)

    ref_probs = [c / ref_total for c in reference_counts]
    prod_probs = [c / prod_total for c in production_counts]

    # Compute TVD
    tvd = 0.5 * sum(abs(p - q) for p, q in zip(ref_probs, prod_probs))

    # Drift detection
    drift_detected = tvd > threshold

    return {
        "score": float(tvd),
        "drift_detected": drift_detected
    }