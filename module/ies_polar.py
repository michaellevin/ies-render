import numpy as np
from collections import namedtuple

PolarCoordinates = namedtuple("PolarCoordinates", ["r", "theta"])


class IESPolar:
    """
    The class is used to convert cartesian coordinates to polar coordinates
    and vice versa
    """

    INVALID_POLAR_COORDINATES = PolarCoordinates(-1, -1)

    def __init__(self, size):
        self.center = size / 2

    def cartesian2polar(
        self,
        x: int,
        y: int,
        pixel_size: float = None,
        top_border: int = None,
        bottom_border: int = None,
    ):
        """Convert cartesian coordinates to polar coordinates

        Args:
            x (int): X coordinate of image (0, 0) is top left corner of image
            y (int): Y coordiante
            pixel_size (float, optional): If set, converts radius from pixels to meters. Defaults to None.
            top_border (int, optional): If set, the upper border of black area. Defaults to None.
            bottom_border (int, optional): If set, the lower border of black area. Defaults to None.

        Returns:
            PolarCoordinates: namedtuple with radius (in pixels or meters) and angle (in degrees)

        Scheme:
        (0,0)           180°
        +----------------*----------------+---> X
        |                |                |
        |                |                |
        |                |                |
        |                |                |
        |                |                |
        |                |                |
        |--------------- 0 ---------------|
        |                |\               |
        |                | \              |
        |                |θ°\             |
        |                |   \ r          |
        |                |    * (x,y)     |
        |                |                |
        +----------------+----------------+ (size, size)
        |                0°
        |
        Y
        """
        dx = x - self.center
        if top_border is None:
            dy = -y + self.center
        else:
            # * if height != 0, area of light's size is black
            if top_border < y < bottom_border:
                return self.INVALID_POLAR_COORDINATES
            elif y <= top_border:
                dy = -y + top_border
            elif y >= bottom_border:
                dy = -y + bottom_border

        # * if central point
        if x == y == self.center:
            return PolarCoordinates(0, 0)

        # * any other point
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.degrees(np.arctan2(dy, dx))  # Convert to degrees
        if theta < 0:
            theta += 360  # Ensure theta is in [0, 360)
        theta = (theta + 90) % 360  # Adjust so that bottom center is 0 degrees
        if theta > 180:
            theta -= 360  # Adjust so that theta is in [-180, 180)

        if pixel_size:
            r *= pixel_size
        return PolarCoordinates(r, theta)

    def polar2cartesian(self, polar: PolarCoordinates):
        ...
