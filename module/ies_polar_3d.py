import numpy as np
from collections import namedtuple
from math import sqrt, atan2, degrees, acos

PolarCoordinates3D = namedtuple("PolarCoordinates3D", ["r", "theta", "phi"])


# def point3d2polar(point: tuple[float, float, float]):
#     """Calculate point 3D polar coordinates in IES format"""
#     x, y, z = point
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = (np.degrees(np.arctan2(y, x)) + 360) % 360  # Normalize to 0-360°
#     phi = np.degrees(np.arccos(z / r)) if r != 0 else 0  # Avoid division by zero
#     return PolarCoordinates3D(r, theta, phi)


# def point3d2polar(point: tuple[float, float, float]):
#     """Calculate point 3D polar coordinates in IES format."""
#     x, y, z = point

#     # Calculate radial distance
#     r = sqrt(x**2 + y**2 + z**2)

#     # Calculate raw angle in the X-Y plane
#     raw_theta = degrees(atan2(y, x))  # Range: -180° to 180°
#     raw_theta = (raw_theta + 360) % 360  # Normalize to 0° to 360°

#     # Adjust theta based on X-axis separation
#     if x >= 0:
#         theta = raw_theta  # 0° to 180° for positive X
#     else:
#         theta = 360 - raw_theta  # 180° to 360° for negative X

#     # Calculate phi (vertical angle)
#     phi = degrees(acos(z / r)) if r != 0 else 0  # Avoid division by zero

#     return PolarCoordinates3D(r, theta, phi)


def point3d2polar(point: tuple[float, float, float]):
    """Calculate point 3D polar coordinates in IES format."""
    x, y, z = point

    # Calculate radial distance
    r = sqrt(x**2 + y**2 + z**2)

    # Calculate raw angle in the X-Y plane
    raw_theta = degrees(atan2(y, x))  # Range: -180° to 180°
    raw_theta = (raw_theta + 360) % 360  # Normalize to 0° to 360°

    # Adjust theta to ensure it's in the range (0, 180)
    if raw_theta > 180:
        theta = 360 - raw_theta
        # theta = raw_theta - 180
    else:
        theta = raw_theta

    # Calculate phi (vertical angle)
    phi = degrees(acos(z / r)) if r != 0 else 0  # Avoid division by zero

    return PolarCoordinates3D(r, theta, phi)
