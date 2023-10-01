# -*- coding: utf-8 -*-
"""IES render strategy module.

This module contains the RenderStrategy abstract class and its subclasses.
For understanfing how the render works, please refer to the following article:
http://docs.autodesk.com/ACD/2011/ENU/filesAUG/WS73099cc142f48755f058a10f71c104f3-3b1a.htm
OR docs/IES_description.pdf.

You can also look through docs/ies_math_asset.hipnc (SideFX Houdini) file, 
where a kind of IES render setup is implemented.
"""

from abc import ABC, abstractmethod
import typing
from collections import namedtuple
from scipy.interpolate import interp1d
import numpy as np
from PIL import Image

try:
    from .ies_parser import IESData
except ImportError:
    from ies_parser import IESData

RenderAttrs = namedtuple(
    "RenderData",
    [
        "center",
        "pixel_size",
        "center_in_meters",
        "x_values",
        "light_X_axis",
        "light_Y_axis",
        "light_top_border",
        "light_bottom_border",
        "y_start",
        "y_end",
    ],
)


class RenderStrategy(ABC):
    WALL_SIZE = 5  # meters

    def __init__(self, ies_data: IESData):
        self._ies_data = ies_data

    def render(
        self, size: int, horizontal_angle: float = None, distance: float = None
    ) -> Image:
        """
        Render method.

        :param size: The size of the rendered image.
        :param horizontal_angle: The horizontal angle.
        :param distance: The distance from the wall in meters.
        """

        render_attrs = self.prepare_render_data(size)

        image = Image.new("RGB", (size, size))

        candela_lists = list(self._ies_data.candela_values.values())
        candela_array = np.array(candela_lists).T
        light_point = np.array(
            [render_attrs.center_in_meters, render_attrs.center_in_meters, distance]
        )
        horizontal_angles = self.get_horizontal_angles(
            size, horizontal_angle, distance, render_attrs
        )
        # print(f"horizontal_angles: {horizontal_angles}")

        # Initialize an array to store pixel values
        candela_values_all = []
        L_values = np.zeros((size, render_attrs.y_end - render_attrs.y_start))

        # Loop through x to handle non-vectorizable parts (e.g., conditionals, interpolation)
        for x, alpha in enumerate(horizontal_angles):
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

            interpolation = interp1d(
                self._ies_data.vertical_angles,
                candela_values,
                kind="linear",
                fill_value="extrapolate",
            )

            candela_values_all.append(candela_values)
            # Compute all betas and L values at once
            y_values = (
                np.arange(render_attrs.y_start, render_attrs.y_end)
                * render_attrs.pixel_size
            )
            ray_L_XY = (
                np.column_stack(
                    (
                        np.full_like(y_values, render_attrs.x_values[x]),
                        y_values,
                        np.zeros_like(y_values),
                    )
                )
                - light_point
            )

            # Calculate the squared distance (RD) and identify where it's close to zero
            RD = np.sum(ray_L_XY**2, axis=1)
            RD = np.maximum(RD, 0.00001)  # Replace 0 value for not deleting by zero
            nb_y = ray_L_XY / np.sqrt(RD)[:, np.newaxis]
            dot_products_y = np.clip(np.dot(nb_y, render_attrs.light_Y_axis), -1.0, 1.0)
            vertical_angles = np.degrees(np.arccos(dot_products_y))
            # vertical_angles[np.all(ray_L_XY == 0, axis=1)] = 0
            L_dir = interpolation(vertical_angles)
            decay_values = 1 / RD
            L = L_dir * decay_values
            # Compute and store pixel values
            L_values[x, :] = L

        L_max = np.max(candela_values_all)
        pixel_values = np.zeros(
            (size, render_attrs.y_end - render_attrs.y_start), dtype=np.uint8
        )
        for x, alpha in enumerate(horizontal_angles):
            # Normalize L values and convert to pixel values
            # TODO: here we can adjust brightness if needed. L_values are in huge range and we clamp then to 0-1
            pixel_values[x, :] = (255 * np.clip(L_values[x, :] / L_max, 0, 1)).astype(
                np.uint8
            )

        # * Update the image with computed pixel values
        for x in range(size):
            for y in range(render_attrs.y_start, render_attrs.y_end):
                # TODO light size
                # if render_attrs.light_top_border is not None:
                #     if (
                #         render_attrs.light_top_border
                #         < y
                #         < render_attrs.light_bottom_border
                #     ):
                #         continue  # black area
                if x == y == render_attrs.center:
                    continue  # black area
                pixel_value = pixel_values[x, y - render_attrs.y_start]
                image.putpixel((x, y), (pixel_value, pixel_value, pixel_value))

        return image

    def prepare_render_data(self, size: int) -> RenderAttrs:
        center = size // 2
        pixel_size = self.WALL_SIZE / size
        center_in_meters = center * pixel_size
        x_values = (
            np.arange(size) * pixel_size
        )  # array of x coordiantes of image in meters
        light_height = self._ies_data.height
        light_height_in_pixels = light_height / pixel_size
        light_X_axis = np.array([1, 0, 0])
        light_Y_axis = np.array([0, 1, 0])
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
            center,
            pixel_size,
            center_in_meters,
            x_values,
            light_X_axis,
            light_Y_axis,
            light_top_border,
            light_bottom_border,
            y_start,
            y_end,
        )

    @abstractmethod
    def get_horizontal_angles(
        self, size, horizontal_angle, distance, render_attrs
    ) -> np.ndarray:
        """Getting horizontal angles

        :param size (int): The size of the rendered image.
        :param horizontal_angle (float): The horizontal angle.
        :param distance (float): The distance from the wall in meters.
        :param render_attrs (RenderAttrs): namedtuple of pre-calculated render attributes

        To be implemented by subclasses.
        """
        pass


# Implement the RenderStrategy for each scenario
class Render0(RenderStrategy):
    """
    The luminaire is assumed to be laterally
    symmetric in all planes.
    """

    def get_horizontal_angles(self, size, *args, **kwargs) -> np.ndarray:
        return np.zeros(size)


class Render0_90(RenderStrategy):
    """
    the luminaire is assumed to be symmetric in
    each quadrant
    """

    def get_horizontal_angles(
        self, size, horizontal_angle, distance, render_attrs
    ) -> np.ndarray:
        if distance == 0.0:
            # Directly assign alphas based on x position relative to light source
            horizontal_angles = np.where(
                render_attrs.x_values >= render_attrs.center_in_meters,
                horizontal_angle,
                90 - horizontal_angle,
            )
        else:
            # Compute all alphas at once using vectorized operations
            light_point = np.array(
                [render_attrs.center_in_meters, render_attrs.center_in_meters, distance]
            )
            x_projection_points = np.column_stack(
                (
                    render_attrs.x_values,
                    np.full(size, render_attrs.center_in_meters),
                    np.zeros(size),
                )
            )

            # Compute all ray_L_Xproj at once
            ray_L_Xproj = x_projection_points - light_point

            na = render_attrs.light_X_axis / np.sqrt(
                np.dot(render_attrs.light_X_axis, render_attrs.light_X_axis)
            )
            nb = ray_L_Xproj / np.sqrt(np.sum(ray_L_Xproj**2, axis=1))[:, np.newaxis]
            dot_products = np.clip(np.dot(nb, na), -1.0, 1.0)
            horizontal_angles = np.degrees(np.arccos(dot_products))

        return horizontal_angles


class Render0_180(RenderStrategy):
    """
    The distribution is symmetric about a vertical plane:
    left plane angles are 180 - H
    """

    def get_horizontal_angles(
        self, size, horizontal_angle, distance, render_attrs
    ) -> np.ndarray:
        if distance == 0.0:
            # Directly assign alphas based on x position relative to light source
            horizontal_angles = np.where(
                render_attrs.x_values >= render_attrs.center_in_meters,
                horizontal_angle,
                180 - horizontal_angle,
            )
        else:
            # Compute all alphas at once using vectorized operations
            light_point = np.array(
                [render_attrs.center_in_meters, render_attrs.center_in_meters, distance]
            )
            x_projection_points = np.column_stack(
                (
                    render_attrs.x_values,
                    np.full(size, render_attrs.center_in_meters),
                    np.zeros(size),
                )
            )

            # Compute all ray_L_Xproj at once
            ray_L_Xproj = x_projection_points - light_point

            na = render_attrs.light_X_axis / np.sqrt(
                np.dot(render_attrs.light_X_axis, render_attrs.light_X_axis)
            )
            nb = ray_L_Xproj / np.sqrt(np.sum(ray_L_Xproj**2, axis=1))[:, np.newaxis]
            dot_products = np.clip(np.dot(nb, na), -1.0, 1.0)
            horizontal_angles = np.degrees(np.arccos(dot_products))

        return horizontal_angles


class Render0_360(RenderStrategy):
    """
    the luminaire is assumed to
    exhibit no lateral symmetry.
    """

    def get_horizontal_angles(
        self, size, horizontal_angle, distance, render_attrs
    ) -> np.ndarray:
        if distance == 0.0:
            # Directly assign alphas based on x position relative to light source
            horizontal_angles = np.where(
                render_attrs.x_values >= render_attrs.center_in_meters,
                horizontal_angle,
                180 + horizontal_angle,
            )
        else:
            # Compute all alphas at once using vectorized operations
            light_point = np.array(
                [render_attrs.center_in_meters, render_attrs.center_in_meters, distance]
            )
            x_projection_points = np.column_stack(
                (
                    render_attrs.x_values,
                    np.full(size, render_attrs.center_in_meters),
                    np.zeros(size),
                )
            )

            # Compute all ray_L_Xproj at once
            ray_L_Xproj = x_projection_points - light_point

            na = render_attrs.light_X_axis / np.sqrt(
                np.dot(render_attrs.light_X_axis, render_attrs.light_X_axis)
            )
            nb = ray_L_Xproj / np.sqrt(np.sum(ray_L_Xproj**2, axis=1))[:, np.newaxis]
            dot_products = np.clip(np.dot(nb, na), -1.0, 1.0)
            horizontal_angles = np.degrees(np.arccos(dot_products))

        return horizontal_angles
