import os
from collections import deque, namedtuple
import math


class BrokenIESFileError(Exception):
    """Exception raised for errors in the IES file format."""

    def __init__(self, message="IES file is broken"):
        self.message = message
        super().__init__(self.message)


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


class IES_Parser:
    """_summary_
    Eager parsing IES file and return IESData namedtuple
    """

    def __init__(self, ies_path: str):
        self._ies_path = ies_path
        if self._ies_path and os.path.exists(self._ies_path):
            self._ies_data = self._parse()
        else:
            raise FileNotFoundError("IES file not found")

    def _parse(self) -> IESData:
        def _parse_line(line: str) -> deque:
            return deque(map(float, line.split()))

        with open(self._ies_path, "r") as f:
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
            # hor_str = (
            #     str(self.ies_data.horizontal_angles)
            #     + " "
            #     + str(len(self.ies_data.horizontal_angles))
            # )
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
    ies_path = "examples/vertical_angles.ies"
    ies = IES_Parser(ies_path)
    print(ies)
