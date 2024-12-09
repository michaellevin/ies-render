import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from module import ies_calc


def test_calculate_3d(x, y):
    ies_path = r"examples\SDL-PA1-20-722-U-T2U-HSS-A_1377 lumens.ies"
    point = (
        x,
        y,
        6,
    )
    h_offset = 0.114  # 3
    orient = 45
    luminance = ies_calc.ies_calculate_luminance_at_point(
        ies_path, h_offset, orient, point
    )
    print(f"luminance: {luminance}")


def test_calculate_3d_box_avg():
    ies_path = r"examples\SDL-PA1-20-722-U-T2U-HSS-A_1377 lumens.ies"
    point1 = (-5, -5)  # First corner of the box
    point2 = (5, 5)  # Opposite corner of the box
    mh = 6  # height of luminaire
    h_offset = 0.1143
    orient = -30
    avg_luminance = ies_calc.average_luminance_on_box(
        ies_path,
        point1,
        point2,
        mh,
        h_offset,
        orient,
        step=1,
    )

    # assert avg_luminance == 30.0


test_calculate_3d(x=0, y=0)
test_calculate_3d(x=1, y=1)
test_calculate_3d(x=1, y=-1)
# test_calculate_3d(x = 1, y = 0)
# test_calculate_3d(x = -1, y = 0)
# test_calculate_3d(x = 1, y = 1)
test_calculate_3d_box_avg()
