import typing
from collections import namedtuple
import math
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image, ImageFilter  # , ImageEnhance

try:
    from . import ies_parser
    from . import utils
except ImportError:
    import ies_parser
    import utils

PolarCoordinates = namedtuple("PolarCoordinates", ["r", "theta"])


class IES_Thumbnail_Generator:
    def __init__(self, ies_path: str, h_angle: float = 0, size: int = 512):
        self.ies_path = ies_path
        self.ies_parser = None
        try:
            self.ies_parser = ies_parser.IES_Parser(self.ies_path)
        except (FileNotFoundError, ies_parser.BrokenIESFileError) as e:
            print(e)
        print(self.ies_parser)

        self.wall_size = 5  # meters

        self.size = size
        self.center = size / 2
        self.pixel_size = self.wall_size / self.size  # 5/512 = 0.009m (1 pixel ~ 1cm)

        # if self.ies_parser:
        #     self._generate(h_angle)

    @utils.timing_decorator
    def generate(
        self,
        size: typing.Optional[int] = None,
        horizontal_angle: typing.Optional[float] = None,
        distance: typing.Optional[float] = None,
        blur_radius: typing.Optional[float] = None,
        out_path: typing.Optional[str] = None,
        save: bool = True,
    ):
        if size:
            self.size = size
            self.center = size / 2
            self.pixel_size = self.wall_size / self.size
        # * Preparations
        #   - if height != 0
        light_height = self.ies_parser.ies_data.height
        light_height_in_pixels = light_height / self.pixel_size
        if light_height != 0:
            light_top_border = self.center - light_height_in_pixels / 2
            light_bottom_border = self.center + light_height_in_pixels / 2

        #   - get vertical range
        start_v_angle, end_v_angle = (
            int(self.ies_parser.ies_data.vertical_angles[0]),
            int(self.ies_parser.ies_data.vertical_angles[-1]),
        )
        #  - get horizontal (azimuthal) range
        if start_v_angle == 0 and end_v_angle == 90:  # V ∈ [0, 90]
            # distribution lies completely in the bottom hemisphere
            y_start, y_end = self.size // 2, self.size
        elif start_v_angle == 90 and end_v_angle == 180:  # V ∈ [90, 180]
            # the distribution lies completely in the top hemisphere
            y_start, y_end = 0, self.size // 2
        elif start_v_angle == 0 and end_v_angle == 180:  # V ∈[0, 180]
            # the distribution lies in both hemispheres
            y_start, y_end = 0, self.size

        # * Generate the image
        image = Image.new("RGB", (self.size, self.size))
        axial_symmetry = len(self.ies_parser.ies_data.horizontal_angles) == 1
        if axial_symmetry:  # H = 0
            # the distribution is axially symmetric
            candela_values = self.ies_parser.ies_data.candela_values[0.0]
            L_max = max(candela_values)
            x_start, x_end = self.size // 2, self.size

            interpolation = interp1d(
                self.ies_parser.ies_data.vertical_angles,
                candela_values,
                kind="linear",
                fill_value="extrapolate",
            )

            # print(f"\nX0: {v_start}, X1: {v_end}, Y0: {h_start}, Y1: {h_end}")

            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    if light_height != 0:
                        polar = self._to_polar(
                            x, y, light_top_border, light_bottom_border
                        )
                    else:
                        polar = self._to_polar(x, y)

                    if polar.r == -1:
                        pixel_value = 0
                    else:
                        pixel_value = self._compute_pixel_value(
                            interpolation, polar.r, polar.theta, L_max, D=distance
                        )
                    image.putpixel((x, y), (pixel_value, pixel_value, pixel_value))

            # mirror
            right_half = image.crop((self.size // 2, 0, self.size, self.size))
            left_half = right_half.transpose(Image.FLIP_LEFT_RIGHT)
            image.paste(left_half, (0, 0))

        else:
            # the distribution is not axially symmetric
            end_h_angle = self.ies_parser.ies_data.horizontal_angles[-1]
            if end_h_angle == 90:  # H ∈ [0, 90]
                # the distribution is symmetric in each quadrant
                ...
            elif end_h_angle == 180:  # H ∈ [0, 180]
                # the distribution is symmetric about a vertical plane:
                # left plane angles are 180 - H
                x_start, x_end = 0, self.size
                candela_right_values = self.ies_parser.ies_data.candela_values[
                    float(horizontal_angle)
                ]
                candela_left_values = self.ies_parser.ies_data.candela_values[
                    180 - float(horizontal_angle)
                ]
                # print(candela_right_values)
                # print(candela_left_values)
                interpolation_right = interp1d(
                    self.ies_parser.ies_data.vertical_angles,
                    candela_right_values,
                    kind="linear",
                    fill_value="extrapolate",
                )
                max_right_value = max(candela_right_values)
                interpolation_left = interp1d(
                    self.ies_parser.ies_data.vertical_angles,
                    candela_left_values,
                    kind="linear",
                    fill_value="extrapolate",
                )
                max_left_value = max(candela_left_values)
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        polar = self._to_polar(x, y)
                        pixel_value = self._compute_pixel_value(
                            interpolation_right
                            if polar.theta >= 0
                            else interpolation_left,
                            polar.r,
                            math.fabs(polar.theta),
                            max_right_value if polar.theta >= 0 else max_left_value,
                            D=distance,
                        )
                        image.putpixel((x, y), (pixel_value, pixel_value, pixel_value))

            elif end_h_angle > 180:  # H ∈ [0, <=360]
                # the distribution exhibits no lateral symmetries
                x_start, x_end = 0, self.size
                ...

        # if brightness != 1:
        #     enhancer = ImageEnhance.Brightness(image)
        #     image = enhancer.enhance(brightness)

        # * Blur a bit
        if blur_radius:
            image = image.filter(ImageFilter.GaussianBlur(blur_radius))

        # * Save the image to a file
        if save:
            if not out_path:
                out_path = self.ies_path.replace(
                    ".ies", f"_s{size}_d{distance}_h{horizontal_angle}.png"
                )
            image.save(out_path)

        return image

    def _to_polar(
        self,
        x: int,
        y: int,
        light_top_border: typing.Optional[int] = None,
        light_bottom_border: typing.Optional[int] = None,
    ) -> PolarCoordinates:
        # * if height != 0
        if light_top_border is not None:
            if light_top_border < y < light_bottom_border:
                return PolarCoordinates(-1, -1)
            else:
                if y < light_top_border:
                    dy = -y + light_top_border
                elif y > light_bottom_border:
                    dy = -y + light_bottom_border
        # * if central point
        if x == self.center and y == self.center:
            return PolarCoordinates(0, 0)
        # * any other point
        dx = x - self.center
        if light_top_border is None:
            dy = -y + self.center
        r = np.sqrt(dx * dx + dy * dy)
        # theta = np.degrees(np.arctan2(-dx, dy))  # Convert to degrees
        theta = np.degrees(np.arctan2(dy, dx))  # Convert to degrees
        if theta < 0:
            theta += 360  # Ensure theta is in [0, 360)
        theta = (theta + 90) % 360  # Adjust so that bottom center is 0 degrees
        if theta > 180:
            theta -= 360  # Adjust so that theta is in [-180, 180)
        return PolarCoordinates(r, theta)

    def _compute_pixel_value(
        self,
        interpolation: typing.Callable[[float], float],
        r: float,  # radius from the center in pixels
        theta: float,  # vertical angle in degrees: 0° in bottom, 180° in top
        L_max: float,  # max lumen intensity in candelas
        D: float,  # Distance from the light source to the wall in meters
    ) -> int:  # color value  (grayscale)
        # * calculate decay value
        R = r * self.pixel_size  # translate radius from pixels to meters
        # * calculate luminance interpolated value, distance D against the wall
        if D != 0.0:
            RD = np.sqrt(R * R + D * D)
            decay_value = 1 / (RD * RD) if R != 0 else 1

            t = np.sqrt((R * np.sin(np.radians(theta))) ** 2 + D**2)
            beta = np.degrees(np.arcsin(t / RD))
            if theta > 90:
                beta = 180 - beta
            # print(theta, beta)
            L_dir = interpolation(beta)  # candela value for this ray (vertical angle)
        else:
            decay_value = 1 / (R * R) if R != 0 else 1
            L_dir = interpolation(theta)

        # * calculate decaued luminance
        L = L_dir * decay_value

        # Brgihtness control
        # ...

        # * adjust simple linear tone maping
        color = 255 * L / L_max

        # Gamma correction
        # color = 255 * math.pow(color / 255, 1 / 2.2)

        return int(color)
