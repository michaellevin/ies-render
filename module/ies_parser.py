import os
from collections import deque, namedtuple
import math
import platform


class BrokenIESFileError(Exception):
    """Exception raised for errors in the IES file format."""

    def __init__(self, message="IES file is broken"):
        self.message = message
        super().__init__(self.message)


def get_next_numbers(f, count):
    """
    Retrieve a specified amount of numeric values from a file stream.

    This function reads lines from the provided file stream, splits them
    into whitespace-separated strings, and accumulates the numeric values.
    It continues to read lines until the specified amount of numbers is
    collected or the file ends. If the file ends before enough numbers
    are gathered, it raises a BrokenIESFileError.

    Args:
        f (file): A file stream object opened in text mode.
        count (int): The number of numeric values to retrieve from the file stream.

    Returns:
        list[str]: A list containing `count` string representations of the numbers.

    Raises:
        BrokenIESFileError: If the end of file is reached before `count` numbers are retrieved.
    """
    numbers = []
    while len(numbers) < count:
        line = next(f, None)
        if line is None:
            raise BrokenIESFileError("Unexpected end of file while reading numbers")
        numbers.extend(line.split())
    return numbers[:count]


IESData = namedtuple(
    "IESData",
    [
        "vertical_angles",  # list of vertical angles
        "horizontal_angles",  # list of horizontal angles
        "candela_values",  # {horizontal_angle: [candela_values]}
        "max_value",  # max value of candela_values (float)
        "num_lamps",  # number of lamps (int)
        "lumens_per_lamp",  # lumens per lamp (float)
        "multiplier",  # multiplier (float)
        "width",  # width (float)
        "length",  # length (float)
        "height",  # height (float)
        "shape",  # shape (str)
    ],
)


class IES_Parser:
    """
    Eager parsing IES file
    Returns IESData namedtuple
    """

    def __init__(self, ies_path: str):
        self._ies_path = ies_path
        if self._ies_path and os.path.exists(self._ies_path):
            self._ies_data = self._parse()
        else:
            raise FileNotFoundError("IES file not found")

    def _parse(self) -> IESData:
        def _parse_line(line: str) -> deque:
            cleaned_line = line.replace(",", " ")
            return deque(map(float, cleaned_line.split()))

        with open(
            self._ies_path,
            "r",
            encoding="Windows-1252" if platform.system() != "Windows" else None,
        ) as f:
            for line in f:
                if line.strip() == "TILT=NONE":
                    break

            # * Get sizes and other data (13 numbers)
            light_data = get_next_numbers(f, 13)  # f.readline().split()
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
            # TODO (all types of shapes)
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
                candela_values.extend(_parse_line(f.readline()))
                if len(candela_values) == num_vertical_angles * num_horizontal_angles:
                    break

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

    @property
    def ies_data(self) -> IESData:
        return self._ies_data

    def __repr__(self) -> str:
        if not self._ies_data:
            return "Broken file"

        bold = "\033[1m"
        underline = "\033[4m"
        red = "\033[91m"  # Red color
        green = "\033[92m"  # Green color
        yellow = "\033[93m"  # Yellow color
        blue = "\033[94m"  # Blue color
        reset = "\033[0m"
        message = f"IES file: {underline}{blue}{self._ies_path}{reset}\n"
        message += f"{bold}Shape:\t{self._ies_data.shape}, L={self._ies_data.length}m, H={self._ies_data.height}m{reset}\n"
        vert_str = f"{self._ies_data.vertical_angles[0]}, {self._ies_data.vertical_angles[1]}, ... {self._ies_data.vertical_angles[-1]} [{len(self.ies_data.vertical_angles)} values]\n"
        message += f"{bold}{underline}{green}Vertical:{reset}\n\t" + vert_str

        if len(self._ies_data.horizontal_angles) == 1:
            hor_str = f"{self._ies_data.horizontal_angles[0]}\n"
            message += f"{bold}{underline}{green}Horizontal:{reset}\n\t" + hor_str
            message += f"{bold}{underline}{green}Candela:{reset}\n\t" + ", ".join(
                map(str, self._ies_data.candela_values[0.0])
            )
        else:
            hor_str = f"{self._ies_data.horizontal_angles[0]}, {self._ies_data.horizontal_angles[1]}, ... {self._ies_data.horizontal_angles[-1]} [{len(self.ies_data.horizontal_angles)} values]\n"
            message += f"{bold}{underline}{green}Horizontal:{reset}\n\t" + hor_str

            message += f"{bold}{underline}{green}Candela:{reset}\n"
            start_h = self._ies_data.horizontal_angles[0]
            end_h = self._ies_data.horizontal_angles[-1]
            message += f"\t{bold}{yellow}{int(start_h)}:{reset}\t" + ", ".join(
                map(str, self._ies_data.candela_values[start_h])
            )
            message += f"\n\t{bold}...{reset}\n"
            message += f"\n\t{bold}{yellow}{int(end_h)}:{reset}\t" + ", ".join(
                map(str, self._ies_data.candela_values[end_h])
            )
            message += "\n"
        return message

    def __call__(self) -> IESData:
        return self._ies_data


if __name__ == "__main__":
    ies_path = "examples/ies-lights-pack/defined-diffuse-spot.ies"
    ies_path = "examples/ies-lights-pack/star-focused.ies"
    # ies_path = "examples/horiz_angles.ies"
    ies = IES_Parser(ies_path)
    print(ies)
