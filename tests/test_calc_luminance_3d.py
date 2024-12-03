import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from module import ies_calc


def test_calculate_3d():
    ies_path = r"examples\SDL-PA1-20-722-U-T2U-HSS-A_1377 lumens.ies"
    point = (4, 4, 6)
    luminance = ies_calc.ies_calculate_luminance_at_point(ies_path, point)
    print(f"luminance: {luminance}")
    # assert luminance == 30.0


def test_calculate_3d_box_avg():
    ies_path = r"examples\SDL-PA1-20-722-U-T2U-HSS-A_1377 lumens.ies"
    point1 = (-5, -5)  # First corner of the box
    point2 = (5, 5)  # Opposite corner of the box
    height = 6  # Height of the box surface
    avg_luminance = ies_calc.average_luminance_on_box(
        ies_path,
        point1,
        point2,
        height,
        # num_points=4
        step=2,
    )
    print(f"avg_luminance: {avg_luminance}")
    # assert avg_luminance == 30.0


# test_calculate_3d()
test_calculate_3d_box_avg()
