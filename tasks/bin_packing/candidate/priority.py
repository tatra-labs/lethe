import numpy as np


def priority(item, bins):
    """Best fit: prefer the bin that the item fills most tightly."""
    return -(bins - item)
