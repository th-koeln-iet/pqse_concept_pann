import os
import pickle
from copy import deepcopy

import numpy as np
from scipy import stats

from tools import SplitComplexConverter
from tools.split_complex import SplitComplexMode


def read_data(data_path, network_data, input_converter: SplitComplexConverter = None,
              target_converter: SplitComplexConverter = None, transpose_data_format=False):
    network_data['data_length_in_min'] = 527026
    network_data['step_width_in_min'] = 15
    network_data['input_converter'] = input_converter
    network_data['target_converter'] = target_converter
    network_data = read_network_data(data_path, network_data, transpose_data_format=transpose_data_format)
    scaling_axes = (0, 1) if network_data['data_format'] == 'channels_last' else (0, 3)
    scale_data(network_data, axes=scaling_axes, sep_output_scaler=True)
    return network_data


def mask_data(data, feature_indices, feature_axis):
    """
    Apply the mask to the input data and reshape the result.
    :param data: The input data array.
    :param feature_indices: Indices of the features to be masked.
    :param feature_axis: Axis of the features.
    :return: Reshaped masked data array.
    """
    # if indexing is negative, convert it to positive
    feature_axis = feature_axis if feature_axis >= 0 else feature_axis + data.ndim

    # Construct a tuple for advanced indexing
    index_tuple = tuple(feature_indices if axis == feature_axis else slice(None) for axis in range(data.ndim))

    # Extract the features along the specified axis using advanced indexing
    masked_data = data[index_tuple]
    return masked_data


def read_network_data(data_path, network_data, transpose_data_format=True):
    """
    read network data from file path, expects the following structure:
    data_path
        grid_name
            y_train.pic
            y_test.pic
            y_validation.pic
            y_mats_per_frequency.pic

    :param data_path: path to the data
    :param network_data: dictionary
    :return: dictionary with additional keys: x_train, y_train, x_test, y_test, x_val, y_val, input_dims, output_dims, ...
    """
    data_path = os.path.join(data_path, network_data['grid_name'])
    input_converter = network_data['input_converter']
    target_converter = network_data['target_converter']
    mask_axis = 1 if network_data['data_format'] == "channels_last" else -1
    harmonic_axis = 1 if network_data['data_format'] == "channels_first" else -1
    # Dataset is split up in train test and validation set
    with open(os.path.join(data_path, 'y_train.pic'), 'rb') as pic:
        train = pickle.load(pic)
    train_np, frequencies = convert_to_numpy(train)
    if transpose_data_format:
        train_np = np.transpose(train_np, (0, 2, 1))
    network_data['frequencies'] = frequencies
    train_np = fix_phase_angle_problem(train_np, harmonic_axis=harmonic_axis, harmonics=[2, 5, 8, 11, 14, 17])
    x_train = mask_data(train_np, network_data['mask_x_test'], mask_axis)
    network_data['x_train'] = input_converter.convert(x_train)
    network_data['y_train'] = target_converter.convert(train_np)
    network_data['input_dims'] = network_data['x_train'].shape[1:]
    network_data['output_dims'] = network_data['y_train'].shape[1:]
    with open(os.path.join(data_path, 'y_test.pic'), 'rb') as pic:
        test = pickle.load(pic)
    test_np, _ = convert_to_numpy(test)
    if transpose_data_format:
        test_np = np.transpose(test_np, (0, 2, 1))
    test_np = fix_phase_angle_problem(test_np, harmonic_axis=harmonic_axis, harmonics=[2, 5, 8, 11, 14, 17])
    x_test = mask_data(test_np, network_data['mask_x_test'], mask_axis)
    network_data['x_test'] = input_converter.convert(x_test)
    network_data['y_test'] = target_converter.convert(test_np)
    with open(os.path.join(data_path, 'y_validation.pic'), 'rb') as pic:
        validation = pickle.load(pic)
    val_np, _ = convert_to_numpy(validation)
    if transpose_data_format:
        val_np = np.transpose(val_np, (0, 2, 1))
    val_np = fix_phase_angle_problem(val_np, harmonic_axis=harmonic_axis, harmonics=[2, 5, 8, 11, 14, 17])
    x_val = mask_data(val_np, network_data['mask_x_test'], mask_axis)
    network_data['x_val'] = input_converter.convert(x_val)
    network_data['y_val'] = target_converter.convert(val_np)

    # read admittance matrices from file
    with open(os.path.join(data_path, 'y_mats_per_frequency.pic'), 'rb') as pic:
        y_mats = pickle.load(pic)

        network_data['Y_matrices_per_frequency'] = convert_y_mats(y_mats, data_format=network_data['data_format'],
                                                                  modes=input_converter.modes)
    return network_data


def add_measurement_noise(x, noise_level):
    """
    Add measurement noise to the voltage measurements
    :param x: voltage measurements
    :param noise_level: standard deviation of the noise
    :return: x with added noise
    """
    return x + np.random.normal(0, noise_level, x.shape)


def convert_to_numpy(list_of_dicts):
    """
    Convert a list of dictionaries to a numpy array.
    :param list_of_dicts:
    :return: multidimensional numpy array
    """
    # Assuming the list is non-empty and all dictionaries have the same keys
    if not list_of_dicts:
        raise ValueError("The list of dictionaries is empty.")

    # Extract and save the keys (frequencies)
    frequencies = list(list_of_dicts[0].keys())

    # Pre-allocate a list to hold all arrays
    all_arrays = []

    # Iterate over each dictionary in the list
    for dct in list_of_dicts:
        # Verify that the keys are consistent
        if list(dct.keys()) != frequencies:
            raise ValueError("Inconsistent keys in dictionaries.")

        # Stack the arrays from this dictionary
        current_array = np.stack([dct[freq] for freq in frequencies], axis=0)
        all_arrays.append(current_array)

    # Convert the list of arrays to a numpy array
    numpy_array = np.array(all_arrays)

    return numpy_array, frequencies


def scale_data(network_data, axes=(0, 3), sep_output_scaler=True):
    """
    Scale the data using the scaler defined in network_data.
    Different scalers are used only if the parameter is set or if the shape mismatches on axes other than the specified ones.
    This happens only if the features of input and target have a different shape (e.g. mixed cartesian / polar input and cartesian output)
    :param network_data: dictionary
    :param axes: tuple, axes along which the scaling is to be consistent
    :param sep_output_scaler: if True uses a separate scaler for the output
    (required if output shape on the non-specified axes does not match input shape,
    e.g. if input consists of polar and cartesian coordinates while input consists only of cartesian)
    :return: network_data with scaled data and scalers for input and output
    """
    scaler = network_data['scaler']
    network_data['x_train_n'] = scaler.fit_transform(network_data['x_train'], axes)
    network_data['x_test_n'] = scaler.transform(network_data['x_test'])
    network_data['x_val_n'] = scaler.transform(network_data['x_val'])

    # Determine if shapes mismatch on non-specified axes
    shape_mismatch = False
    input_shape = list(network_data['x_train'].shape)
    output_shape = list(network_data['y_train'].shape)
    for axis in set(range(max(len(input_shape), len(output_shape)))) - set(axes):
        # Adjust index out of range for shorter shape
        if axis >= len(input_shape) or axis >= len(output_shape) or input_shape[axis] != output_shape[axis]:
            shape_mismatch = True
            break

    # Use different scaler for output if shape mismatches on non-specified axes
    if shape_mismatch or sep_output_scaler:
        # save the previous scaler along with internal parameters as input scaler
        network_data['input_scaler'] = deepcopy(scaler)
        network_data['y_train_n'] = scaler.fit_transform(network_data['y_train'], axes)
    else:
        network_data['y_train_n'] = scaler.transform(network_data['y_train'])
    network_data['y_test_n'] = scaler.transform(network_data['y_test'])
    network_data['y_val_n'] = scaler.transform(network_data['y_val'])
    return network_data


def fix_phase_angle_problem(arr: np.ndarray, harmonic_axis=2, harmonics=[2, 5, 8, 11, 14, 17], threshold=1e-9,
                            angle_cutoff=0.4):
    """
    Phase angles cannot be measured if the magnitude is zero or close to zero.
    This results in outliers in the complex plots which the neural network has difficulties estimating.
    This function fixes the problem by replacing the phase angle with the most common phase angle per frequency.
    :param arr: numpy array of complex numbers
    :param harmonic_axis: axis along which the harmonics are stored
    :param harmonics: list of harmonic frequencies to be processed, 0-indexed harmonic --> 2 means 3rd harmonic
    :param threshold: threshold below which magnitudes are considered to be zero
    :param angle_cutoff: cutoff value for angles in radians. Any phase value for magnitudes equal to zero outside
    of the cutoff range will be set to the mean angle.
    :return: Modified array
    """
    # move harmonic axis to the end
    arr = np.moveaxis(arr, harmonic_axis, -1)

    # Loop through the specific frequencies
    for f in harmonics:
        slice_harm = arr[..., f]
        magnitude = np.abs(slice_harm)
        phase = np.angle(slice_harm)
        mode_angle = stats.mode(phase.ravel())[0]
        mag_near_zero_indices = np.abs(magnitude) < threshold

        phase_values_for_mag_near_zero = np.extract(mag_near_zero_indices, phase)
        node_indices = np.argwhere(mag_near_zero_indices)

        unique_nodes = np.unique(node_indices[:, -1])
        for node in unique_nodes:
            filtered_indices = node_indices[node_indices[:, -1] == node]

            condition = np.abs(phase_values_for_mag_near_zero[filtered_indices[:, 0]] - mode_angle) > angle_cutoff
            phase[filtered_indices[condition, 0], filtered_indices[condition, 1]] = mode_angle

        # convert magnitude and phase back to complex number
        arr[..., f] = magnitude * np.exp(1j * phase)
    # move harmonic axis back to original position
    arr = np.moveaxis(arr, -1, harmonic_axis)
    return arr


def convert_y_mats(y_mats: dict, data_format="channels_last", modes=SplitComplexMode.CARTESIAN):
    """
    Get the admittance matrices for the given frequencies as numpy array.
    :param y_mats: dictionary of admittance matrices per frequency
    :param data_format: "channels_last" or "channels_first"
    :param modes: SplitComplexMode
    :return: numpy array of admittance matrices converted to the desired format
    """
    # convert to numpy
    y_mats_np = np.array([np.array(value) for key, value in y_mats.items()])
    # convert to split complex
    sc_conv = SplitComplexConverter(modes=modes, target_axis=3)
    y_mats_cart = sc_conv.convert(y_mats_np)
    if data_format == 'channels_first':
        return np.swapaxes(y_mats_cart, 1, 3)
    return np.swapaxes(y_mats_cart, 0, 2)
