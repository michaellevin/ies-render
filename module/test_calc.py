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
