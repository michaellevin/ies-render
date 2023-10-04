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
            # logging.info(ies_parser)
            self._ies_data = ies_parser.ies_data
            return self._ies_data
        except (FileNotFoundError, BrokenIESFileError) as err:
            logging.error(err)
            return None

    def _create_render_strategy(self, ies_data: IESData) -> RenderStrategy:
        horizontal_angle_last = int(ies_data.horizontal_angles[-1])
        if horizontal_angle_last == 0:
            # the luminaire is assumed to be laterally symmetric in all planes
            # return Render0(ies_data)
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

        # * Rendering
        image = self.render_strategy.render(
            size=size, horizontal_angle=horizontal_angle, distance=distance
        )

        # * Post-effects: blur
        if blur_radius:
            image = image.filter(ImageFilter.BoxBlur(blur_radius))

        # * Save
        if save:
            if not out_path:
                out_path = self.ies_path.replace(
                    ".ies", f"_s{size}_d{distance}_h{horizontal_angle}.png"
                )
            image.save(out_path)
            logging.info(f" Saved image to {out_path}")

        return image
