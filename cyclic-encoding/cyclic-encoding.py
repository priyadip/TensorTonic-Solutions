import math

def cyclic_encoding(values: list, period: float) -> list:
    """
    Returns the sine and cosine encoding of every cyclic value.
    """
    return [
        [
            math.sin(2 * math.pi * v / period),
            math.cos(2 * math.pi * v / period)
        ]
        for v in values
    ]