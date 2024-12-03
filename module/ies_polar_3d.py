import numpy as np
from collections import namedtuple

PolarCoordinates3D = namedtuple("PolarCoordinates3D", ["r", "theta", "phi"])


def point3d2polar(point: tuple[float, float, float]):
    """Calculate point 3D polar coordinates in IES format"""
    x, y, z = point
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = (np.degrees(np.arctan2(y, x)) + 360) % 360  # Normalize to 0-360°
    phi = np.degrees(np.arccos(z / r)) if r != 0 else 0  # Avoid division by zero
    return PolarCoordinates3D(r, theta, phi)
