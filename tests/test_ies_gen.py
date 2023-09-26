import numpy as np
from module import IES_Thumbnail_Generator


def test_to_polar():
    instance = IES_Thumbnail_Generator(ies_path=None, size=512)
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
        polar = instance._to_polar(*coords)
        assert np.isclose(
            polar.theta, expected_theta, atol=0.5
        ), f"Failed for coords {coords}: expected {expected_theta}, got {polar.theta}"
        print(f"Passed for coords {coords}")


# You can then run this test function, or better yet, use a test runner like pytest.
