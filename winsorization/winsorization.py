import math

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds using linear interpolation.
    """
    n = len(values)
    sorted_vals = sorted(values)

    def percentile(p):
        # Compute interpolation index
        k = (n - 1) * p / 100.0
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return float(sorted_vals[lo])
        return (
            sorted_vals[lo]
            + (k - lo) * (sorted_vals[hi] - sorted_vals[lo])
        )

    lower = percentile(lower_pct)
    upper = percentile(upper_pct)

    # Clip while preserving original order
    return [float(max(lower, min(upper, x))) for x in values]
