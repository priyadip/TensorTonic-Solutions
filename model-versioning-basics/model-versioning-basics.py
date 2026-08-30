def promote_model(models: list) -> str:
    """
    Returns the model name as a string.
    """
    selected = max(
        models,
        key=lambda m: (m["accuracy"], -m["latency"], m["timestamp"])
    )

    return selected["name"]