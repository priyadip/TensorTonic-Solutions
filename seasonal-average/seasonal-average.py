def seasonal_average(series: list, period: int) -> list:
    """
    Returns the average for each position in the seasonal cycle.
    """
    averages = []

    for p in range(period):
        values = series[p::period]
        averages.append(float(sum(values) / len(values)))

    return averages