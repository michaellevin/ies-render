import numpy as np

horizonthal_angles = [0.0, 5.0, 10.0]
candela_dct = {0.0: [30, 20, 10], 5.0: [32, 28, 16], 10.0: [36, 24, 20]}

candela_array = np.array(list(candela_dct.values())).T
alpha = 7.5
candela_values = np.array(
    [np.interp(alpha, horizonthal_angles, row) for row in candela_array]
)
print(f"interpolated: {candela_values}")
print(f"max: {max(candela_values)}")


# # a light is distanced by D meters from the wall
# render_attrs = self.get_render_prepare_data(size)
# center_in_meters = render_attrs.center * render_attrs.pixel_size

# candela_lists = list(self._ies_data.candela_values.values())
# candela_array = np.array(candela_lists).T
# light_point = np.array([center_in_meters, center_in_meters, distance])
# light_X_axis = np.array([1, 0, 0])
# light_Y_axis = np.array([0, 1, 0])

# for x in range(0, size):
#     x_projection_point = np.array(
#         [x * render_attrs.pixel_size, center_in_meters, 0]
#     )
#     ray_L_Xproj = x_projection_point - light_point
#     alpha = angle_between(light_X_axis, ray_L_Xproj)

#     # Interpolate the candela values for the target angle
#     if alpha in self._ies_data.horizontal_angles:
#         candela_values = self._ies_data.candela_values[alpha]
#     else:
#         candela_values = np.array(
#             [
#                 np.interp(alpha, self._ies_data.horizontal_angles, row)
#                 for row in candela_array
#             ]
#         )

#     interpolation = interp1d(
#         self._ies_data.vertical_angles,
#         candela_values,
#         kind="linear",
#         fill_value="extrapolate",
#     )
#     L_max = max(candela_values)
#     for y in range(render_attrs.y_start, render_attrs.y_end):
#         ray_L_XY = np.array(
#             [x * render_attrs.pixel_size, y * render_attrs.pixel_size, 0]
#             - light_point
#         )
#         beta = angle_between(light_Y_axis, ray_L_XY)
#         RD = np.dot(ray_L_XY, ray_L_XY)  # square # np.linalg.norm(ray_L_XY)
#         L_dir = interpolation(beta)  # candela value for this ray (vertical angle)
#         decay_value = 1 / RD
#         L = L_dir * decay_value
#         pixel_value = int(255 * L / L_max)
#         image.putpixel((x, y), (pixel_value, pixel_value, pixel_value))
