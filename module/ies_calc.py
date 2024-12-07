"""Calculate ies candelas at point"""

import numpy as np
from .ies_parser import IES_Parser, IESData
from .ies_polar_3d import point3d2polar
import math
from pprint import pprint
import matplotlib.pyplot as plt


def bilinear_interpolation(r, theta, phi, IESData):
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
    # Edge case handling for vertical angle
    if vertical_angle <= vertical_angles[0]:
        v1 = v2 = vertical_angles[0]
    elif vertical_angle >= vertical_angles[-1]:
        v1 = v2 = vertical_angles[-1]
    else:
        v1 = max(a for a in vertical_angles if a <= vertical_angle)
        v2 = min(a for a in vertical_angles if a >= vertical_angle)
    Q11 = candela_values[h1][vertical_angles.index(v1)]
    Q12 = candela_values[h1][vertical_angles.index(v2)]
    Q21 = candela_values[h2][vertical_angles.index(v1)]
    Q22 = candela_values[h2][vertical_angles.index(v2)]

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

    luminance = math.cos(math.radians(phi)) * P / (r**2)
    # print(f"Inputs: r={r}, theta={theta}, phi={phi}")
    # print(f"Horizontal angles: {horizontal_angles}")
    # print(f"Vertical angles: {vertical_angles}")
    # print(f"Interpolation weights: wh1={wh1}, wh2={wh2}, wv1={wv1}, wv2={wv2}")
    # print(f"Candela values: Q11={Q11}, Q12={Q12}, Q21={Q21}, Q22={Q22}")
    return luminance


def calculate_luminance(
    ies_data: IESData,
    point: tuple[float, float, float],
    h_offset: float,
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
    x, y, z = point
    z -= h_offset
    polar = point3d2polar((x, y, z))
    print(polar)
    luminance = bilinear_interpolation(
        polar.r,
        polar.theta,
        polar.phi,
        ies_data,
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


def average_luminance_on_box(ies_path, point1, point2, mh, h_offset, step=1):
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

    points_x = []
    points_y = []
    luminance_values = []

    total_luminance = 0
    num_points = 0

    for x in x_coords:
        for y in y_coords:
            if x1 <= x <= x2 and y1 <= y <= y2:  # Check if point is inside the box
                point = (x, y, mh)
                luminance = calculate_luminance(ies_data, point, h_offset)
                print(f"Point: {point[0], point[1]}, Luminance: {luminance}")
                total_luminance += luminance
                points_x.append(x)
                points_y.append(y)
                luminance_values.append(luminance)
                num_points += 1

    average_luminance = total_luminance / num_points if num_points > 0 else 0
    print(f"avg_luminance: {average_luminance}")

    # Plot luminance distribution
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(points_x, points_y, c=luminance_values, cmap="viridis", s=50)
    plt.colorbar(scatter, label="Luminance (cd/m²)")
    # plt.title("Luminance Distribution on Box Surface")
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.grid(True)

    # Add labels to each point
    for x, y, lum in zip(points_x, points_y, luminance_values):
        plt.text(x, y, f"{lum:.4f}", fontsize=8, ha="center", va="bottom")

    # Add average luminance to the plot
    plt.text(
        0.5,
        1.05,
        f"Average Luminance: {average_luminance:.4f} cd/m²",
        fontsize=12,
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.show()

    return average_luminance
