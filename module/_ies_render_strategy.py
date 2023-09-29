from abc import ABC, abstractmethod
import typing
from collections import namedtuple
from scipy.interpolate import interp1d
import numpy as np
from PIL import Image

try:
    # from . import ies_parser
    from .utils import timing_decorator
    from .ies_polar import IESPolar, PolarCoordinates
except ImportError:
    from utils import timing_decorator
    from ies_polar import IESPolar, PolarCoordinates

RenderAttrs = namedtuple(
    "RenderData",
    [
        "center",
        "pixel_size",
        "light_top_border",
        "light_bottom_border",
        "y_start",
        "y_end",
    ],
)


def nearest_angle(alpha, angle_list):
    nearest = None
    min_diff = float("inf")

    for angle in angle_list:
        diff = (angle - alpha + 180) % 360 - 180
        if abs(diff) < abs(min_diff):
            min_diff = diff
            nearest = angle

    return nearest


class RenderStrategy(ABC):
    WALL_SIZE = 5  # meters

    def __init__(self, ies_data):
        self._ies_data = ies_data

    @abstractmethod
    def render(self, size: int, horizontal_angle: float = None, distance: float = None):
        """
        Abstract render method to be implemented by subclasses.

        :param size: The size of the rendered image.
        :param horizontal_angle: The horizontal angle.
        :param distance: The distance from the wall in meters.
        :param blur_radius:  The blur radius for the render in pixels.
        """
        pass

    def get_render_prepare_data(self, size: int):
        center = size // 2
        pixel_size = self.WALL_SIZE / size
        light_height = self._ies_data.height
        light_height_in_pixels = light_height / pixel_size
        light_top_border = None
        light_bottom_border = None
        if light_height != 0:
            light_top_border = center - light_height_in_pixels / 2
            light_bottom_border = center + light_height_in_pixels / 2

        start_v_angle, end_v_angle = (
            self._ies_data.vertical_angles[0],
            self._ies_data.vertical_angles[-1],
        )
        if start_v_angle == 0 and end_v_angle == 180:  # V ∈ [0, 180]
            # the distribution lies in both hemispheres
            y_start, y_end = 0, size
        elif start_v_angle == 0 and end_v_angle == 90:  # V ∈ [0, 90]
            # distribution lies completely in the bottom hemisphere
            y_start, y_end = size // 2, size
        elif start_v_angle == 90 and end_v_angle == 180:  # V ∈ [90, 180]
            # the distribution lies completely in the top hemisphere
            y_start, y_end = 0, size // 2

        return RenderAttrs(
            center, pixel_size, light_top_border, light_bottom_border, y_start, y_end
        )


# Implement the RenderStrategy for each scenario
class Render0(RenderStrategy):
    def render(self, size, horizontal_angle, distance):
        """
        Render for case when there is only one horizontal angle = 0.
        This means axial symmetry.

        :param horizontal_angle: Not used for this render strategy.
        """
        render_attrs = self.get_render_prepare_data(size)
        candela_values = self._ies_data.candela_values[0.0]
        L_max = max(candela_values)
        interp_theta_luminance = interp1d(
            self._ies_data.vertical_angles,
            candela_values,
            kind="linear",
            fill_value="extrapolate",
        )
        coords = IESPolar(size)

        # * Generate right half of the image
        image = Image.new("RGB", (size, size))
        for x in range(render_attrs.center, size):
            for y in range(render_attrs.y_start, render_attrs.y_end):
                if self._ies_data.height != 0:
                    polar = coords.cartesian2polar(
                        x,
                        y,
                        render_attrs.pixel_size,
                        top_border=render_attrs.light_top_border,
                        bottom_border=render_attrs.light_bottom_border,
                    )
                else:
                    polar = coords.cartesian2polar(
                        x, y, pixel_size=render_attrs.pixel_size
                    )

                if polar == IESPolar.INVALID_POLAR_COORDINATES:
                    pixel_value = 0
                else:
                    pixel_value = self._compute_pixel_color(
                        polar=polar,
                        interp_func=interp_theta_luminance,
                        L_max=L_max,
                        D=distance,
                    )

                image.putpixel((x, y), (pixel_value, pixel_value, pixel_value))

        # * Mirror right to left
        right_half = image.crop((render_attrs.center, 0, size, size))
        left_half = right_half.transpose(Image.FLIP_LEFT_RIGHT)
        image.paste(left_half, (0, 0))

        return image

    def _compute_pixel_color(
        self,
        polar: IESPolar,
        interp_func: typing.Callable,
        L_max: float,
        D: float = 0.0,
    ):
        # * calculate decay value
        R = polar.r

        # * calculate luminance interpolated value, distance D against the wall
        if D:
            RD = np.sqrt(R * R + D * D)
            decay_value = 1 / (RD * RD) if R != 0 else 1

            t = np.sqrt((R * np.sin(np.radians(polar.theta))) ** 2 + D**2)
            beta = np.degrees(np.arcsin(t / RD))
            if polar.theta > 90:
                beta = 180 - beta
            # print(theta, beta)
            L_dir = interp_func(beta)  # candela value for this ray (vertical angle)
        else:
            decay_value = 1 / (R * R) if R != 0 else 1
            L_dir = interp_func(polar.theta)

        # * calculate decaued luminance
        L = L_dir * decay_value

        # * adjust simple linear tone maping
        color = 255 * L / L_max

        return int(color)


class Render0_90(RenderStrategy):
    def render(self, size, horizontal_angle, distance):
        # Implement rendering for [0, 90] angle range
        ...


class Render0_180(RenderStrategy):
    def render(self, size, horizontal_angle, distance):
        # the distribution is symmetric about a vertical plane:
        # left plane angles are 180 - H
        render_attrs = self.get_render_prepare_data(size)

        coords = IESPolar(size)
        image = Image.new("RGB", (size, size))

        if not distance:
            # D = 0
            nearest_available_horizontal_angle = nearest_angle(
                float(horizontal_angle), self._ies_data.horizontal_angles
            )
            opposite_avaialble_horizontal_angle = nearest_angle(
                180 - float(horizontal_angle), self._ies_data.horizontal_angles
            )
            candela_right_values = self._ies_data.candela_values[
                nearest_available_horizontal_angle
            ]
            candela_left_values = self._ies_data.candela_values[
                opposite_avaialble_horizontal_angle
            ]
            interpolation_right = interp1d(
                self._ies_data.vertical_angles,
                candela_right_values,
                kind="linear",
                fill_value="extrapolate",
            )
            interpolation_left = interp1d(
                self._ies_data.vertical_angles,
                candela_left_values,
                kind="linear",
                fill_value="extrapolate",
            )
            L_max = max(candela_left_values + candela_right_values)

            for x in range(0, size):
                for y in range(render_attrs.y_start, render_attrs.y_end):
                    if self._ies_data.height != 0:
                        polar = coords.cartesian2polar(
                            x,
                            y,
                            render_attrs.pixel_size,
                            top_border=render_attrs.light_top_border,
                            bottom_border=render_attrs.light_bottom_border,
                        )
                    else:
                        polar = coords.cartesian2polar(
                            x, y, pixel_size=render_attrs.pixel_size
                        )

                    if polar.theta >= 0:
                        interp_func = interpolation_right
                    else:
                        interp_func = interpolation_left
                    decay_value = 1 / (polar.r * polar.r) if polar.r != 0 else 1
                    L_dir = interp_func(np.abs(polar.theta))
                    L = L_dir * decay_value
                    pixel_value = int(255 * L / L_max)
                    image.putpixel((x, y), (pixel_value, pixel_value, pixel_value))

        else:
            # a light is distanced by D meters from the wall

            candela_lists = list(self._ies_data.candela_values.values())
            candela_array = np.array(candela_lists).T

            # nearest_available_horizontal_angle = nearest_angle(
            #     float(horizontal_angle), self._ies_data.horizontal_angles
            # )
            # opposite_avaialble_horizontal_angle = nearest_angle(
            #     180 - float(horizontal_angle), self._ies_data.horizontal_angles
            # )
            # candela_right_values = self._ies_data.candela_values[
            #     nearest_available_horizontal_angle
            # ]
            # candela_left_values = self._ies_data.candela_values[
            #     opposite_avaialble_horizontal_angle
            # ]
            # interpolation_right = interp1d(
            #     self._ies_data.vertical_angles,
            #     candela_right_values,
            #     kind="linear",
            #     fill_value="extrapolate",
            # )
            # interpolation_left = interp1d(
            #     self._ies_data.vertical_angles,
            #     candela_left_values,
            #     kind="linear",
            #     fill_value="extrapolate",
            # )
            # L_max = max(candela_left_values + candela_right_values)
            # print(nearest_available_horizontal_angle, candela_right_values)
            # print(opposite_avaialble_horizontal_angle, candela_left_values)

            for x in range(0, size):
                x_ = x - render_attrs.center
                alpha = 90 - np.degrees(
                    np.arctan((x_ * render_attrs.pixel_size) / distance)
                )  # horizontal angle

                # Interpolate the candela values for the target angle
                if alpha in self._ies_data.horizontal_angles:
                    candela_values = self._ies_data.candela_values[alpha]
                else:
                    candela_values = np.array(
                        [
                            np.interp(alpha, self._ies_data.horizontal_angles, row)
                            for row in candela_array
                        ]
                    )

                # print(interpolated_candela_values)
                interpolation = interp1d(
                    self._ies_data.vertical_angles,
                    candela_values,
                    kind="linear",
                    fill_value="extrapolate",
                )
                L_max = max(candela_values)
                # print(
                #     (x, x_),
                #     (x_ * render_attrs.pixel_size) / distance,
                #     np.degrees(np.arctan((x_ * render_attrs.pixel_size) / distance)),
                #     f"horizonthal angle: {alpha}",
                #     f"L max: {L_max}",
                #     f"L: {candela_values[0]}, {candela_values[1]}, {candela_values[2]} .. {candela_values[-1]}",
                # )
                for y in range(render_attrs.y_start, render_attrs.y_end):
                    if self._ies_data.height != 0:
                        polar = coords.cartesian2polar(
                            x,
                            y,
                            render_attrs.pixel_size,
                            top_border=render_attrs.light_top_border,
                            bottom_border=render_attrs.light_bottom_border,
                        )
                    else:
                        polar = coords.cartesian2polar(
                            x, y, pixel_size=render_attrs.pixel_size
                        )

                    RD = np.sqrt(polar.r * polar.r + distance * distance)
                    y_ = y - render_attrs.center
                    beta = np.degrees(np.arccos((y_ * render_attrs.pixel_size) / RD))
                    print(
                        "    ",
                        (x, y),
                        beta,
                    )
                    # if polar.theta >= 0:
                    #     interpolation = interpolation_right
                    # else:
                    #     interpolation = interpolation_left
                    L_dir = interpolation(
                        beta
                    )  # candela value for this ray (vertical angle)
                    decay_value = 1 / RD**2
                    L = L_dir * decay_value
                    pixel_value = int(255 * L / L_max)

                    image.putpixel((x, y), (pixel_value, pixel_value, pixel_value))

        return image


class Render0_360(RenderStrategy):
    def render(self, size, horizontal_angle, distance):
        # Implement rendering for [0, 360] angle range
        ...
