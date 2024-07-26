from enum import Enum

import numpy as np


class SplitComplexMode(Enum):
    CARTESIAN = 0
    POLAR = 1
    EXPONENTIAL = 2


class SplitComplexConverter:
    def __init__(self, modes, target_axis=-1):
        """
        Initialize the converter with a list of modes, the first mode will be at the first two positions in target axis,
        the second mode will be at the next two positions in target axis, and so on.
        :param modes:
        :return:
        """
        if not isinstance(modes, list):
            if isinstance(modes, SplitComplexMode):
                modes = [modes]
            else:
                raise TypeError("modes must be SplitComplexModes or a list of SplitComplexModes")
        self.modes = modes
        self.target_axis = target_axis
        self.titles = []
        self.set_attributes()

    def set_attributes(self):
        for mode in self.modes:
            if mode == SplitComplexMode.CARTESIAN:
                self.titles.append('real')
                self.titles.append('imaginary')
            elif mode == SplitComplexMode.POLAR:
                self.titles.append('magnitude')
                self.titles.append('phase')
            elif mode == SplitComplexMode.EXPONENTIAL:
                self.titles.append('signed magnitude')
                self.titles.append(r"imag($e^{j*\phi}$)")

    def convert(self, data: np.ndarray) -> np.ndarray:
        """
        Convert a multidimensional array according to initialized modes
        :param data: numpy array of complex data
        :return: numpy array of converted data with added dimension at target axis
        """
        # Add axis at the end of the array
        data_shape = data.shape
        res = np.zeros(data_shape + (2 * len(self.modes),))
        i = 0
        for mode in self.modes:
            if mode == SplitComplexMode.CARTESIAN:
                res[..., i] = data.real
                res[..., i + 1] = data.imag
            elif mode == SplitComplexMode.POLAR:
                res[..., i] = np.abs(data)
                res[..., i + 1] = np.angle(data, False)
            elif mode == SplitComplexMode.EXPONENTIAL:
                exp = np.exp(1j * np.angle(data))
                sign_real = np.sign(exp.real)
                res[..., i] = np.abs(data) * sign_real
                res[..., i + 1] = exp.imag
            else:
                raise ValueError("Unknown mode")
            i = i + 2
        # Move target axis to the specified position
        res = np.moveaxis(res, -1, self.target_axis)
        return res


def cartesian_to_polar(arr: np.ndarray, axis: int = 2, real_index=0,
                       imag_index=1, angle_type: str = 'radians'):
    """
    Convert a nD array from Cartesian coordinates (real, imaginary) to polar coordinates (magnitude, phase).
    Indices not chosen in real_index and imag_index will be kept as they are.
    Parameters:
    arr (numpy.ndarray): Input nD array with Cartesian coordinates.
    axis (int): Axis along which the conversion is to be performed. Default is 2.
    real_index (int|tuple[int]): Index along which the real part is located. Default is 0.
    imag_index (int|tuple[int]): Index along which the imaginary part is located. Default is 1.

    Returns:
    numpy.ndarray: Output nD array with polar coordinates (magnitude, phase).
    """
    if angle_type not in ['radians', 'degrees']:
        raise ValueError("angle_type must be 'radians' or 'degrees'")
    arr = arr.astype(np.float64)  # ensure the array is treated as float
    # Swap axes to bring the specified axis to the last dimension
    arr_swapped = np.swapaxes(arr, axis, -1)
    polar_arr_swapped = np.copy(arr_swapped)

    # Extract real and imaginary parts
    real = arr_swapped[..., real_index]
    imag = arr_swapped[..., imag_index]

    # Compute the magnitude and phase
    magnitude = np.sqrt(np.square(real) + np.square(imag))
    phase = np.arctan2(imag, real)
    if angle_type == 'degrees':
        phase = np.degrees(phase)
    # Place magnitude and phase in their new positions
    polar_arr_swapped[..., real_index] = magnitude
    polar_arr_swapped[..., imag_index] = phase

    # Swap axes back to original configuration
    polar_arr = np.swapaxes(polar_arr_swapped, axis, -1)

    return polar_arr


def polar_to_cartesian(arr, axis=2, mag_index=0, phase_index=1,
                       angle_type: str = 'radians'):
    """
    Convert a nD array from polar coordinates (magnitude, phase) to Cartesian coordinates (real, imaginary),
    keeping other elements along the specified axis.

    Parameters:
    arr (numpy.ndarray): Input nD array with polar coordinates.
    axis (int): Axis along which the conversion is to be performed. Default is 2.
    mag_index (int|tuple[int]): Index along which the magnitude is located. Default is 0.
    phase_index (int|tuple[int]): Index along which the phase is located. Default is 1.
    angle_type (str): Type of angle provided. 'radians' | 'degrees'.
    Returns:
    numpy.ndarray: Output nD array with Cartesian coordinates (real, imaginary) and other existing elements.
    """
    if angle_type not in ['radians', 'degrees']:
        raise ValueError("angle_type must be 'radians' or 'degrees'")
    arr = arr.astype(np.float64)  # ensure the array is treated as float

    # Swap axes to bring the specified axis to the last dimension
    arr_swapped = np.swapaxes(arr, axis, -1)
    cartesian_arr_swapped = np.copy(arr_swapped)

    # Extract magnitude and phase
    magnitude = arr_swapped[..., mag_index]
    phase = arr_swapped[..., phase_index]
    if angle_type == 'degrees':
        phase = np.radians(phase)
    # Compute the real and imaginary components
    real_part = magnitude * np.cos(phase)
    imaginary_part = magnitude * np.sin(phase)

    # Place real and imaginary components in their new positions
    cartesian_arr_swapped[..., mag_index] = real_part
    cartesian_arr_swapped[..., phase_index] = imaginary_part

    # Swap axes back to original configuration
    cartesian_arr = np.swapaxes(cartesian_arr_swapped, axis, -1)

    return cartesian_arr


def polar_to_complex(arr: np.ndarray, mag_index=0, phase_index=1, target_axis=-1, degrees=False):
    """
    Convert a polar array with
    :param arr:
    :param target_axis: axis of complex components
    :param mag_index: index of magnitude
    :param phase_index: index of phase
    :param degrees: if True angle type is in degrees, else radians
    :return: complex-valued array
    """
    magnitude = np.take(arr, mag_index, axis=target_axis)
    angle = np.take(arr, phase_index, axis=target_axis)
    if degrees:
        angle = np.radians(angle)
    complex_array = magnitude * np.exp(1j * angle)
    return complex_array
