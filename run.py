import os
import typing
import time
from collections import deque, namedtuple
import sys
import math
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image, ImageEnhance, ImageFilter

# from qtpy.QtGui import QImage, QColor, QPixmap
# from qtpy.QtCore import Qt
# from qtpy.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


class BrokenIESFileError(Exception):
    """Exception raised for errors in the IES file format."""

    def __init__(self, message="IES file is broken"):
        self.message = message
        super().__init__(self.message)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.6f} seconds to run.")
        return result

    return wrapper


IESData = namedtuple(
    "IESData",
    [
        "vertical_angles",
        "horizontal_angles",
        "candela_values",
        "max_value",
        "num_lamps",
        "lumens_per_lamp",
        "multiplier",
        "width",
        "length",
        "height",
        "shape",
    ],
)
PolarCoordinates = namedtuple("PolarCoordinates", ["r", "theta"])


class IES_Parser:
    def __init__(self, ies_path: str):
        self.ies_path = ies_path
        if ies_path and os.path.exists(self.ies_path):
            self.ies_data = self._parse()
        else:
            raise FileNotFoundError("IES file not found")

    # @timing_decorator
    def _parse(self) -> IESData:
        def _parse_line(line: str) -> deque:
            return deque(map(float, line.split()))

        with open(self.ies_path, "r") as f:
            for line in f:
                if line.strip() == "TILT=NONE":
                    break

            # * Get sizes and other data
            light_data = f.readline().split()
            num_lamps = int(light_data[0])
            lumens_per_lamp = float(light_data[1])
            multiplier = float(light_data[2])
            num_vertical_angles = int(light_data[3])
            num_horizontal_angles = int(light_data[4])
            unit = int(light_data[6])  # 1 - feet, 2 - meters
            k = 1 if unit == 2 else 0.3048
            width = float(light_data[7]) * k
            length = float(light_data[8]) * k
            height = float(light_data[9]) * k
            # TODO
            if all(i == 0 for i in [width, length, height]):
                shape = "point"
            elif height == 0 and width < 0 and width == length:
                shape = "circular"
            elif height == 0 and width < 0 and width != length:
                shape = "ellipse"
            elif height != 0 and width < 0 and width == length:
                shape = "vertical cylinder"
            elif height != 0 and width != length and length < 0:
                shape = "vertical ellipsoidal cylinder"
            elif height < 0 and width == length == height:
                shape = "sphere"
            elif height < 0 and width < 0 and length < 0:
                shape = "ellipsoidal spheroid"
            elif height == 0:
                shape = "rectangular"
            else:
                shape = "rectangular with luminous sides"

            # Skip the next (only one) line
            next(f)

            # * Read vertical angles
            vertical_angles, horizontal_angles, candela_values = (
                deque(),
                deque(),
                deque(),
            )

            while True:
                vertical_angles.extend(_parse_line(f.readline()))
                if len(vertical_angles) == num_vertical_angles:
                    break
            vertical_angles = list(vertical_angles)
            if int(vertical_angles[0]) not in [0, 90, -90] or int(
                vertical_angles[-1]
            ) not in [90, 180]:
                raise BrokenIESFileError()

            # * Read horizontal angles
            while True:
                horizontal_angles.extend(_parse_line(f.readline()))
                if len(horizontal_angles) == num_horizontal_angles:
                    break
            horizontal_angles = list(horizontal_angles)
            if int(horizontal_angles[0]) not in [0, -90] or int(
                horizontal_angles[-1]
            ) not in [0, 90, 180, 360]:
                raise BrokenIESFileError()

            # * Read candela values
            while True:
                next_line = f.readline()
                if not next_line:
                    break  # End of file
                candela_values.extend(_parse_line(next_line))
            candela_values = list(candela_values)
            max_value = max(candela_values)

            # * assert len(vert_angles)*len(horizontal_angles) == len(candelas)
            if len(vertical_angles) * len(horizontal_angles) != len(candela_values):
                raise BrokenIESFileError()

            V = len(candela_values) // len(horizontal_angles)
            candela_values_dct = {
                n: candela_values[i * V : (i + 1) * V]
                for i, n in enumerate(horizontal_angles)
            }

            return IESData(
                vertical_angles,
                horizontal_angles,
                candela_values_dct,
                max_value,
                num_lamps,
                lumens_per_lamp,
                multiplier,
                math.fabs(width),
                math.fabs(length),
                math.fabs(height),
                shape,
            )

    def __repr__(self) -> str:
        bold = "\033[1m"
        underline = "\033[4m"
        red = "\033[91m"  # Red color
        green = "\033[92m"  # Green color
        yellow = "\033[93m"  # Yellow color
        blue = "\033[94m"  # Blue color
        reset = "\033[0m"
        message = f"IES file: {underline}{blue}{self.ies_path}{reset}\n"
        message += f"{bold}Shape:\t{self.ies_data.shape}, L={self.ies_data.length}m, H={self.ies_data.height}m{reset}\n"
        vert_str = f"{self.ies_data.vertical_angles[0]}, {self.ies_data.vertical_angles[1]}, ... {self.ies_data.vertical_angles[-1]} [{len(self.ies_data.vertical_angles)} values]\n"
        message += f"{bold}{underline}{green}Vertical:{reset}\n\t" + vert_str

        if len(self.ies_data.horizontal_angles) == 1:
            hor_str = f"{self.ies_data.horizontal_angles[0]}\n"
            message += f"{bold}{underline}{green}Horizontal:{reset}\n\t" + hor_str
            message += f"{bold}{underline}{green}Candela:{reset}\n\t" + ", ".join(
                map(str, self.ies_data.candela_values[0.0])
            )
        else:
            hor_str = f"{self.ies_data.horizontal_angles[0]}, {self.ies_data.horizontal_angles[1]}, ... {self.ies_data.horizontal_angles[-1]} [{len(self.ies_data.horizontal_angles)} values]\n"
            # hor_str = (
            #     str(self.ies_data.horizontal_angles)
            #     + " "
            #     + str(len(self.ies_data.horizontal_angles))
            # )
            message += f"{bold}{underline}{green}Horizontal:{reset}\n\t" + hor_str

            message += f"{bold}{underline}{green}Candela:{reset}\n"
            start_h = self.ies_data.horizontal_angles[0]
            end_h = self.ies_data.horizontal_angles[-1]
            message += f"\t{bold}{yellow}{int(start_h)}:{reset}\t" + ", ".join(
                map(str, self.ies_data.candela_values[start_h])
            )
            message += f"\n\t{bold}...{reset}\n"
            message += f"\n\t{bold}{yellow}{int(end_h)}:{reset}\t" + ", ".join(
                map(str, self.ies_data.candela_values[end_h])
            )
        return message


class IES_Thumbnail_Generator:
    def __init__(self, ies_path: str, h_angle: float = 0, size: int = 512):
        self.ies_path = ies_path
        self.ies_parser = None
        try:
            self.ies_parser = IES_Parser(self.ies_path)
        except (FileNotFoundError, BrokenIESFileError) as e:
            print(e)

        self.size = size
        self.center = size / 2
        self.wall_size = 5  # meters
        self.pixel_size = self.wall_size / self.size  # 5/512 = 0.009m (1 pixel ~ 1cm)
        print(self.ies_parser)
        # if self.ies_parser:
        #     self._generate(h_angle)

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

    @timing_decorator
    def generate(
        self,
        horizontal_angle: float = 0,
        distance: float = 0,
        out_path: typing.Optional[str] = None,
    ):
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
        radius = 1
        image = image.filter(ImageFilter.GaussianBlur(radius))

        # * Save the image to a file
        if not out_path:
            out_path = self.ies_path.replace(".ies", "_py.png")
        image.save(out_path)


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


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super(MainWindow, self).__init__()

#         self.setWindowTitle("Polar Coordinates Visualization")

#         widget = QWidget(self)
#         layout = QVBoxLayout(widget)
#         self.label = QLabel()
#         layout.addWidget(self.label)
#         self.setCentralWidget(widget)

#         # Generate QImage
#         image = QImage(512, 512, QImage.Format_ARGB32)

#         for x in range(512):
#             for y in range(512):
#                 # Calculate relative coordinates
#                 dx = x - 255.5
#                 dy = y - 255.5

#                 r = np.sqrt(dx * dx + dy * dy)
#                 theta = np.degrees(np.arctan2(-dx, dy))  # Convert to degrees
#                 # if theta < 0:  # Adjust to [0, 180] range
#                 #     theta += 180

#                 # Compute the pixel value based on decay and interpolation
#                 pixel_value = compute_pixel_value(r, math.fabs(theta))
#                 # print(pixel_value)
#                 # Set the pixel color
#                 image.setPixelColor(x, y, QColor(pixel_value, pixel_value, pixel_value))

#         # Display the image in the QLabel
#         self.label.setPixmap(QPixmap(image))


if __name__ == "__main__":
    # test polar coordiantes
    #  test_to_polar()

    # check ies
    ies_paths = [
        r"C:\Cd\tmp\ABC\ies\vertical_angles.ies",
        r"C:\Cd\tmp\ABC\ies\horiz_angles.ies",
        r"C:\Cd\tmp\ABC\ies\ies-lights-pack\area-light.ies",
    ]
    tb = IES_Thumbnail_Generator(ies_paths[2], size=1024)
    tb.generate(horizontal_angle=0, distance=0.3)

    # app = QApplication(sys.argv)
    # window = MainWindow()
    # window.show()
    # sys.exit(app.exec_())
