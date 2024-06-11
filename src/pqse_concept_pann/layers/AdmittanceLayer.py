import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from typing import Sequence


def match_shapes_by_repeating(arr: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """
    Adjusts the shape of arr by repeating its elements along any axis necessary
    to match the target_shape.

    :param arr: Array to be reshaped.
    :param target_shape: Target shape.
    :return: Reshaped array.
    """
    for axis, (target_size, current_size) in enumerate(zip(target_shape, arr.shape)):
        if current_size != target_size:
            # Calculate the repeat factor for the current dimension
            repeat_factor = target_size // current_size
            arr = np.repeat(arr, repeat_factor, axis=axis)
    return arr


def match_shapes_by_alternating_repeating(arr: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """
    Adjusts the shape of the array by alternately repeating its elements along any axis
    necessary to match the target_shape, resulting in an alternating pattern. e.g. (1,2) -> (1,2,1,2)
    :param arr: array to be reshaped
    :param target_shape: target shape
    :return: reshaped array
    """
    for axis, (target_size, current_size) in enumerate(zip(target_shape, arr.shape)):
        if current_size != target_size:
            # Calculate the repeat factor for the current dimension
            repeat_factor = target_size // current_size
            # Ensure the repeat factor is an integer that exactly divides the target size
            assert target_size % current_size == 0, "Target size must be an integer multiple of current size"

            # Generate an array of indices to create the alternating pattern
            indices = np.tile(np.arange(current_size), repeat_factor)

            # Use advanced indexing to create the alternating pattern
            arr = np.take(arr, indices, axis=axis)

    return arr


def prepare_admittance_matrices(admittance_matrices, known_bus_indices, data_format='channels_last'):
    """
    Modify the given admittance matrices based on known bus indices.
    Extend dimensions and compute the absolute value of elements.
    :param admittance_matrices: Array of admittance matrices per frequency and complex component.
    :param known_bus_indices: Indices of buses with known elements (e.g., measurement buses).
                         None if inputs for all nodes are available.
    :param data_format: Format of the input data.
    :return: Array of modified admittance matrices.
    """
    if data_format == 'channels_first':
        y_node_axes = (-2, -1)
    else:
        y_node_axes = (0, 1)
    # Extract elements corresponding to known bus indices from the original admittance matrices.
    if known_bus_indices is not None:
        # move first node axis to the end
        admittance_matrices = np.moveaxis(admittance_matrices, y_node_axes[0], -1)
        pruned_admittance_matrices = admittance_matrices[..., known_bus_indices]
        # move first node axis back to the front
        pruned_admittance_matrices = np.moveaxis(pruned_admittance_matrices, -1, y_node_axes[0])
    else:
        pruned_admittance_matrices = admittance_matrices

    # Ensure that the node axes are positive
    y_node_axes_positive = [axis if axis >= 0 else len(admittance_matrices.shape) + axis for axis in y_node_axes]

    # feature axes are those which are not node axes
    y_feature_axes = tuple([i for i in range(len(admittance_matrices.shape)) if i not in y_node_axes_positive])

    # Compute the absolute value of the pruned admittance matrices.
    abs_admittance_matrices = np.abs(pruned_admittance_matrices)

    reps = [1 for _ in range(len(admittance_matrices.shape))]
    # Extend the dimensions of the absolute value matrices to fit kernel shape
    for i in y_feature_axes:  # for each feature axis
        abs_admittance_matrices = abs_admittance_matrices[..., np.newaxis]  # add a new dimension
        reps.append(abs_admittance_matrices.shape[i])  # remember the shape of feature axis

    # Repeat the extended matrix along the newly added dimensions.
    tiled_matrices = np.tile(abs_admittance_matrices, reps)

    # Reorder the dimensions so that each node axis has features immediately following it.
    # e.g. (node, feature1, feature2, node, feature1, feature2)
    if data_format == 'channels_last':
        reordered_matrices = np.moveaxis(tiled_matrices, y_node_axes_positive[1], -(len(y_feature_axes) + 1))
    else:  # or (feature1, feature2, node, feature1, feature2, node) in case of channels_first
        reordered_matrices = np.moveaxis(tiled_matrices, y_node_axes_positive[1], y_node_axes[1])

    return reordered_matrices


class AdmittanceLayer(kl.Layer):
    def __init__(self, y_mats, mask_indices=None, activation=None, layer_scaling_factor=1,
                 data_format: str = 'channels_last', **kwargs):
        super(AdmittanceLayer, self).__init__(**kwargs)
        self.mask_indices = mask_indices
        self.activation = tf.keras.activations.get(activation)
        self.layer_scaling_factor = layer_scaling_factor
        self.data_format = data_format
        self.y_mats = prepare_admittance_matrices(y_mats, mask_indices, self.data_format)
        self.n_feature_axis = len(y_mats.shape) - 2  # 2 node axes
        if 'weight_initializer' in kwargs:
            self.weight_initializer = kwargs['weight_initializer']
        else:
            self.weight_initializer = "uniform"


    def build(self, input_shape):
        if self.data_format == 'channels_last':
            feature_dimensions = [input_shape[i] for i in range(2, len(input_shape))]  # skip batch and node axes
            # kernel shape is [node, feature1, feature2, node, feature1, feature2]
            kernel_shape = list(input_shape[1:]) + [self.y_mats.shape[-len(input_shape) + 1]] + feature_dimensions
        else:
            feature_dimensions = [input_shape[i] for i in
                                  range(1, len(input_shape) - 1)]  # skip batch (0) and node axes (-1)
            # kernel shape is [feature1, feature2, node, feature1, feature2, node]
            kernel_shape = list(input_shape[1:]) + feature_dimensions + [self.y_mats.shape[-1]]

        # Determine if the layer is the first in the sequence
        self.is_first_layer = len(input_shape) == self.n_feature_axis + 2  # +2 for batch and node axes
        if self.layer_scaling_factor != 1:
            if self.is_first_layer:
                kernel_shape = [self.layer_scaling_factor] + kernel_shape
            tile_shape = list(kernel_shape[:1]) + [1 for _ in range(len(self.y_mats.shape))]
            self.y_mats_tiled = tf.cast(tf.tile(tf.expand_dims(self.y_mats, axis=0), tile_shape), tf.float32)
        else:
            self.y_mats_tiled = self.y_mats

        if kernel_shape != self.y_mats_tiled.shape:  # additional features such as currents on feature axis
            self.y_mats_tiled = match_shapes_by_alternating_repeating(self.y_mats_tiled, kernel_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.weight_initializer,
                                      trainable=True)

        super().build(input_shape)

    def call(self, x):
        masked_weights = tf.multiply(self.kernel, self.y_mats_tiled)
        if self.data_format == 'channels_first':
            if self.layer_scaling_factor == 1:
                output = tf.einsum('bfci,fcijko->bjko', x, masked_weights)
                # (batch, freq, complex, node) * (freq, complex, node, freq, complex, node) --> (batch, freq, complex, node)
            else:
                if self.is_first_layer:
                    output = tf.einsum('bfci,nfcijko->bnjko', x, masked_weights)
                else:
                    output = tf.einsum('bnfci,nfcijko->bnjko', x, masked_weights)
        else:  # channels_last
            if self.layer_scaling_factor == 1:
                output = tf.einsum('bifc,ifcojk->bojk', x, masked_weights)
                # (batch, node, freq, complex) * (node, freq, complex, node, freq, complex) --> (batch, node, freq, complex)
            else:
                if self.is_first_layer:
                    output = tf.einsum('bifc,nifcojk->bnojk', x, masked_weights)
                else:
                    output = tf.einsum('bnifc,nifcojk->bnojk', x, masked_weights)

        if self.activation is not None:
            output = self.activation(output)
        # TODO: for deployment use tensorflow model optimization toolkit to actually prune network
        return output
