import math

def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    """
    Returns a dictionary with the promotion decision and metrics.
    """
    n = len(production_log)

    # Accuracy
    production_accuracy = sum(
        log["prediction"] == log["actual"] for log in production_log
    ) / n

    shadow_accuracy = sum(
        log["prediction"] == log["actual"] for log in shadow_log
    ) / n

    accuracy_gain = shadow_accuracy - production_accuracy

    # Nearest-rank P95
    latencies = sorted(log["latency_ms"] for log in shadow_log)
    p95_index = math.ceil(0.95 * n) - 1
    shadow_latency_p95 = latencies[p95_index]

    # Agreement rate
    agreement_rate = sum(
        p["prediction"] == s["prediction"]
        for p, s in zip(production_log, shadow_log)
    ) / n

    metrics = {
        "shadow_accuracy": shadow_accuracy,
        "production_accuracy": production_accuracy,
        "accuracy_gain": accuracy_gain,
        "shadow_latency_p95": shadow_latency_p95,
        "agreement_rate": agreement_rate,
    }

    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": bool(promote),
        "metrics": metrics,
    }