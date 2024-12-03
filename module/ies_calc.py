"""Calculate ies candelas at point"""

import numpy as np
from .ies_parser import IES_Parser, IESData
from .ies_polar_3d import point3d2polar
import math
from pprint import pprint


def bilinear_interpolation(r, theta, phi, height, IESData):
    """
    Calculates the luminance at a point (r, theta, phi) using bilinear
    interpolation of IES data, with edge case handling.

    Args:
        r: Distance from the light source.
        theta: Azimuth angle in degrees.
        phi: Elevation angle in degrees.
        IESData: Named tuple containing IES data.

    Returns:
        The interpolated luminance value at the given point.
    """

    horizontal_angle = theta
    vertical_angle = phi

    horizontal_angles = IESData.horizontal_angles
    vertical_angles = IESData.vertical_angles
    candela_values = IESData.candela_values

    # Edge case handling for horizontal angle
    if horizontal_angle <= horizontal_angles[0]:
        h1 = h2 = horizontal_angles[0]
    elif horizontal_angle >= horizontal_angles[-1]:
        h1 = h2 = horizontal_angles[-1]
    else:
        h1 = max(a for a in horizontal_angles if a <= horizontal_angle)
        h2 = min(a for a in horizontal_angles if a >= horizontal_angle)
    # print(h1, h2)
    # Edge case handling for vertical angle
    if vertical_angle <= vertical_angles[0]:
        v1 = v2 = vertical_angles[0]
    elif vertical_angle >= vertical_angles[-1]:
        v1 = v2 = vertical_angles[-1]
    else:
        v1 = max(a for a in vertical_angles if a <= vertical_angle)
        v2 = min(a for a in vertical_angles if a >= vertical_angle)
    # print(v1, v2)
    Q11 = candela_values[h1][vertical_angles.index(v1)]
    Q12 = candela_values[h1][vertical_angles.index(v2)]
    Q21 = candela_values[h2][vertical_angles.index(v1)]
    Q22 = candela_values[h2][vertical_angles.index(v2)]
    # print(Q11, Q12, Q21, Q22)
    # Avoid division by zero
    if h2 == h1:
        wh1 = wh2 = 0.5  # Or simply use Q11 (or Q21)
    else:
        wh1 = (h2 - horizontal_angle) / (h2 - h1)
        wh2 = (horizontal_angle - h1) / (h2 - h1)

    if v2 == v1:
        wv1 = wv2 = 0.5  # Or simply use R1 (or R2)
    else:
        wv1 = (v2 - vertical_angle) / (v2 - v1)
        wv2 = (vertical_angle - v1) / (v2 - v1)

    R1 = wh1 * Q11 + wh2 * Q21
    R2 = wh1 * Q12 + wh2 * Q22

    P = wv1 * R1 + wv2 * R2
    # print(P)
    # r -= 0.1156335552861286
    hyp = height * math.cos(math.radians(phi))
    # print(hyp)
    # hyp = 0.0723
    r -= hyp
    luminance = P / (r**2)

    return luminance


def calculate_luminance(
    ies_data: IESData, height: float, point: tuple[float, float, float]
):
    """
    Calculate luminance (candela value) at a 3D point.

    Args:
        point (tuple[float, float, float]): The 3D point (x, y, z).
        ies_data (IESData): Parsed IES data with candela values.

    Returns:
        float: The luminance (candela value) at the given point.
    """
    # Convert point to polar coordinates
    polar = point3d2polar(point)
    print(polar)
    luminance = bilinear_interpolation(
        polar.r, polar.theta, polar.phi, height, ies_data
    )
    return luminance


def ies_calculate_luminance_at_point(
    ies_path: str,
    height: float,
    point: tuple[float, float, float],
):
    """
    Calculate luminance (candela value) at a 3D point.

    Args:
        point (tuple[float, float, float]): The 3D point (x, y, z).
        ies_path (str): Path to the IES file.

    Returns:
        float: The luminance (candela value) at the given point.
    """
    ies_parser = IES_Parser(ies_path)
    ies_data = ies_parser.ies_data
    return calculate_luminance(ies_data, height, point)


# def average_luminance_on_box(ies_path, point1, point2, height, num_points=100):
#     """
#     Calculates the average luminance on a horizontal box surface under an IES light.
#     The box is defined by two diagonally opposite 3D points.

#     Args:
#         IESData: Named tuple containing IES data.
#         point1: First 3D point (x1, y1, z1) defining a corner of the box.
#         point2: Second 3D point (x2, y2, z2) defining the opposite corner of the box.
#         height: Height of the box surface from the light source (this will
#                 override the z-coordinates of point1 and point2).
#         num_points: Number of points to sample for the calculation.

#     Returns:
#         The average luminance on the box surface in cd/m².
#     """
#     ies_parser = IES_Parser(ies_path)
#     ies_data = ies_parser.ies_data

#     x1, y1 = point1  # Ignore z1 from point1
#     x2, y2 = point2  # Ignore z2 from point2

#     x_coords = np.linspace(x1, x2, num_points)
#     y_coords = np.linspace(y1, y2, num_points)

#     total_luminance = 0

#     for x in x_coords:
#         for y in y_coords:
#             print(x, y)
#             polar_coords = point3d2polar((x, y, height))
#             luminance = bilinear_interpolation(
#                 polar_coords.r, polar_coords.theta, polar_coords.phi, ies_data
#             )
#             total_luminance += luminance

#     average_luminance = total_luminance / (num_points**2)
#     return average_luminance


def average_luminance_on_box(ies_path, point1, point2, height, light_z_offset, step=1):
    """
    Calculates the average luminance on a horizontal box surface under an IES light.
    The box is defined by two diagonally opposite 3D points.
    Samples are taken at a specified step size within the box boundaries.

    Args:
        IESData: Named tuple containing IES data.
        point1: First 3D point (x1, y1, z1) defining a corner of the box.
        point2: Second 3D point (x2, y2, z2) defining the opposite corner of the box.
        height: Height of the box surface from the light source (this will
                override the z-coordinates of point1 and point2).
        step: The spacing between sample points in meters.

    Returns:
        The average luminance on the box surface in cd/m².
    """
    ies_parser = IES_Parser(ies_path)
    ies_data = ies_parser.ies_data

    x1, y1 = point1
    x2, y2 = point2

    # Generate coordinates starting from the center
    x_coords = np.arange(0, x2 + step / 2, step)
    x_coords = np.concatenate([-np.flip(x_coords[1:]), x_coords])
    y_coords = np.arange(0, y2 + step / 2, step)
    y_coords = np.concatenate([-np.flip(y_coords[1:]), y_coords])

    total_luminance = 0
    num_points = 0

    for x in x_coords:
        for y in y_coords:
            if x1 <= x <= x2 and y1 <= y <= y2:  # Check if point is inside the box
                polar_coords = point3d2polar((x, y, height))
                luminance = bilinear_interpolation(
                    polar_coords.r,
                    polar_coords.theta,
                    polar_coords.phi,
                    light_z_offset,
                    ies_data,
                )
                total_luminance += luminance  # math.ceil(luminance)
                print(num_points, x, y, luminance)
                num_points += 1

    average_luminance = total_luminance / num_points if num_points > 0 else 0
    return average_luminance
