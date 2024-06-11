import numpy as np
import pandas as pd

from pqse_concept_pann.tools.metrics import mean_absolute_error


def get_mae_split_axis(y_test: np.ndarray, y_pred: np.ndarray, freqs: list, feature_axis: int = 2, error_axis: int = 0):
    """
    Get Mean Absolute Error over all instances for each element in given axis separately
    :param y_pred: prediction result
    :param y_test: ground truth
    :param freqs: frequencies
    :param feature_axis: Axis along which to split the data. Defaults to 2.
    :param error_axis: Axis along which to compute the MAE. Defaults to 0.
    :return list of pandas.DataFrame with MAE for each element in feature_axis
    """
    # move feature axis to last dimension
    y_pred = np.swapaxes(y_pred, feature_axis, -1)
    y_test = np.swapaxes(y_test, feature_axis, -1)
    maes = []
    for i in range(y_pred.shape[-1]):
        maes.append(mean_absolute_error(y_test[..., i], y_pred[..., i], axis=error_axis))
    # move feature axis back to original position
    dfs = []
    for i in range(y_pred.shape[-1]):
        sliced_array = maes[i]
        dataframe = pd.DataFrame(sliced_array, columns=freqs)
        dfs.append(dataframe)
    return dfs


def get_max_error_split_axis(y_test: np.ndarray, y_pred: np.ndarray, freqs: list, feature_axis: int = 2,
                             error_axis: int = 0):
    """
    Get maximum error over all instances for each element in given axis separately
    :param y_pred: prediction result
    :param y_test: ground truth
    :param freqs: frequencies
    :param feature_axis: Axis along which to split the data. Defaults to 2.
    :param error_axis: Axis along which to compute the max error. Defaults to 0.
    :return:
    """
    max_error_n = np.max(np.abs(y_pred - y_test), axis=error_axis)
    dfs = []
    for i in range(max_error_n.shape[feature_axis - 1]):
        sliced_array = max_error_n.take(i, feature_axis - 1)
        dataframe = pd.DataFrame(sliced_array, columns=freqs)
        dfs.append(dataframe)
    return dfs


def get_single_instance_split_axis(array_4d, freqs, axis=2, instance=0):
    """
    Get magnitude and angle of a single instance as pandas dataframe
    :param array_4d: 4 dimensional array with dimensions [batch, frequency, magnitude/angle, node]
    :param freqs: frequencies
    :param instance: int, index of the instance
    :param axis: for each element in axis, the result is returned separately
    :return: magnitude and angle as pandas dataframe
    """
    dfs = []
    for i in range(array_4d.shape[axis]):
        v = array_4d[instance, :, i, :]
        df = pd.DataFrame(v.T, columns=freqs)
        dfs.append(df)
    return tuple(dfs)
