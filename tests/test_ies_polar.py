import numpy as np
from module import IESPolar


def test_to_polar():
    dummy = IESPolar(size=512)
    # Define the expected results
    expected_results = {
        (0, 0): -135.0,
        (255, 0): -179.77,
        (512, 0): 135.0,
        (0, 256): -90.0,
        (256, 256): 0.0,
        (512, 256): 90.0,
        (0, 512): -45.0,
        (256, 512): 0.0,
        (512, 512): 45.0,
    }

    # Test the _to_polar method with different coordinates
    for coords, expected_theta in expected_results.items():
        polar = dummy.cartesian2polar(*coords)
        assert np.isclose(
            polar.theta, expected_theta, atol=0.5
        ), f"Failed for coords {coords}: expected {expected_theta}, got {polar.theta}"
        print(f"Passed for coords {coords}")
