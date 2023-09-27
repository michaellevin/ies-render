import typing
from PIL import ImageFilter  # , ImageEnhance

import logging

logging.basicConfig(level=logging.INFO)

try:
    from .utils import timing_decorator
    from .ies_parser import IES_Parser, IESData, BrokenIESFileError

    # from .ies_polar import IESPolar, PolarCoordinates
    from ._ies_render_strategy import (
        RenderStrategy,
        Render0,
        Render0_90,
        Render0_180,
        Render0_360,
    )
except ImportError:
    from utils import timing_decorator
    from ies_parser import IES_Parser, IESData, BrokenIESFileError

    # from ies_polar import IESPolar, PolarCoordinates
    from _ies_render_strategy import (
        RenderStrategy,
        Render0,
        Render0_90,
        Render0_180,
        Render0_360,
    )


class IES_Thumbnail_Generator:
    def __init__(self, ies_path: str):
        self._ies_path = None
        self._ies_data = None
        self._populate(ies_path)

        if self._ies_data:
            self.render_strategy = self._create_render_strategy(self._ies_data)

    @property
    def ies_path(self):
        return self._ies_path

    @property
    def ies_data(self):
        return self._ies_data

    def _populate(self, ies_path: str) -> IESData | None:
        self._ies_path = ies_path
        try:
            ies_parser = IES_Parser(ies_path)
            logging.info(ies_parser)
            self._ies_data = ies_parser.ies_data
            return self._ies_data
        except (FileNotFoundError, BrokenIESFileError) as err:
            logging.error(err)
            return None

    def _create_render_strategy(self, ies_data: IESData) -> RenderStrategy:
        horizontal_angle_last = int(ies_data.horizontal_angles[-1])
        if horizontal_angle_last == 0:
            # the luminaire is assumed to be laterally symmetric in all planes
            return Render0(ies_data)
        elif horizontal_angle_last == 90:
            # the luminaire is assumed to be symmetric in each quadrant.
            return Render0_90(ies_data)
        elif horizontal_angle_last == 180:
            # the luminaire is assumed to be symmetric about the 0 to 180 degree plane
            return Render0_180(ies_data)
        elif 180 < horizontal_angle_last <= 360:
            # the luminaire is assumed to exhibit no lateral symmetry
            return Render0_360(ies_data)

    @timing_decorator
    def render(
        self,
        size: typing.Optional[int] = 512,
        horizontal_angle: typing.Optional[float] = 0.0,
        distance: typing.Optional[float] = 0.0,
        blur_radius: typing.Optional[float] = 1,
        save: bool = True,
        out_path: typing.Optional[str] = None,
    ):
        logging.info(f" Rendering '{self.ies_path}' with size {size}...")
        image = self.render_strategy.render(
            size=size, horizontal_angle=horizontal_angle, distance=distance
        )

        if blur_radius:
            image = image.filter(ImageFilter.GaussianBlur(blur_radius))

        if save:
            if not out_path:
                out_path = self.ies_path.replace(
                    ".ies", f"_s{size}_d{distance}_h{horizontal_angle}.png"
                )
            image.save(out_path)
            logging.info(f"Saved image to {out_path}")

    """
    @timing_decorator
    def generate(
        self,
        size: typing.Optional[int] = None,
        horizontal_angle: typing.Optional[float] = None,
        distance: typing.Optional[float] = None,
        blur_radius: typing.Optional[float] = None,
        save: bool = True,
        out_path: typing.Optional[str] = None,
    ):
        if size:
            self.size = size
            self.center = size / 2
            self.pixel_size = self.wall_size / self.size

        # * Preparations
        #   - if height != 0
        light_height = self.ies_data.ies_data.height
        light_height_in_pixels = light_height / self.pixel_size
        if light_height != 0:
            light_top_border = self.center - light_height_in_pixels / 2
            light_bottom_border = self.center + light_height_in_pixels / 2

        #   - get vertical range
        start_v_angle, end_v_angle = (
            int(self.ies_data.ies_data.vertical_angles[0]),
            int(self.ies_data.ies_data.vertical_angles[-1]),
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
        axial_symmetry = len(self.ies_data.ies_data.horizontal_angles) == 1
        if axial_symmetry:  # H = 0
            # the distribution is axially symmetric
            candela_values = self.ies_data.ies_data.candela_values[0.0]
            L_max = max(candela_values)
            x_start, x_end = self.size // 2, self.size

            interpolation = interp1d(
                self.ies_data.ies_data.vertical_angles,
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
                            interpolation, polar, L_max, D=distance
                        )
                    image.putpixel((x, y), (pixel_value, pixel_value, pixel_value))

            # mirror
            right_half = image.crop((self.size // 2, 0, self.size, self.size))
            left_half = right_half.transpose(Image.FLIP_LEFT_RIGHT)
            image.paste(left_half, (0, 0))

        else:
            # the distribution is not axially symmetric
            end_h_angle = self.ies_data.ies_data.horizontal_angles[-1]
            if end_h_angle == 90:  # H ∈ [0, 90]
                # the distribution is symmetric in each quadrant
                ...
            elif end_h_angle == 180:  # H ∈ [0, 180]
                # the distribution is symmetric about a vertical plane:
                # left plane angles are 180 - H
                x_start, x_end = 0, self.size
                candela_right_values = self.ies_data.ies_data.candela_values[
                    float(horizontal_angle)
                ]
                candela_left_values = self.ies_data.ies_data.candela_values[
                    180 - float(horizontal_angle)
                ]
                # print(candela_right_values)
                # print(candela_left_values)
                interpolation_right = interp1d(
                    self.ies_data.ies_data.vertical_angles,
                    candela_right_values,
                    kind="linear",
                    fill_value="extrapolate",
                )
                max_right_value = max(candela_right_values)
                interpolation_left = interp1d(
                    self.ies_data.ies_data.vertical_angles,
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
        polar: PolarCoordinates,  # radius from the center in pixels, vertical angle in degrees: 0° in bottom, 180° in top
        L_max: float,  # max lumen intensity in candelas
        D: float,  # Distance from the light source to the wall in meters
    ) -> int:  # color value  (grayscale)
        r = polar.r
        theta = polar.theta

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

    def _candela_at_polar(self, polar):
        r = polar.r
        theta = polar.theta
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

    def _compute_pixel_value_2(
        self,
        polar: PolarCoordinates,  # radius from the center in pixels, vertical angle in degrees: 0° in bottom, 180° in top
        D: float,  # Distance from the light source to the wall in meters
    ) -> int:  # color value  (grayscale)
        r = polar.r
        theta = polar.theta

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
    """
