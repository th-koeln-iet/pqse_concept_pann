import numpy as np
import pandas as pd
import math
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


def calculate_scaling_factor(voltage, base_ref, use_pu):
    """
    Calculate scaling factor based on voltage level and reference base voltage.

    Parameters:
    :param voltage: Voltage level for the bus
    :param base_ref: Reference base voltage for scaling
    :param use_pu: Flag to indicate use of per-unit system

    :return: float: Calculated scaling factor
    """
    if use_pu:
        return base_ref / (voltage / math.sqrt(3))
    else:
        return 1  # No scaling if per-unit system is not used


def calculate_error_metric(y_true, y_pred, orders_to_evaluate, base_kvs, metric='mse', use_pu=False, vt_v=1,
                           excluded_buses=None, min_max_ref=False):
    """
    Calculate Mean Squared Error (MSE), Root Mean Squared Error (RMSE), or Normalized Root Mean Squared Error (NRMSE) per frequency order.

    Parameters:
    :param y_true: True measurement data
    :param y_pred: Predicted or HSE results
    :param orders_to_evaluate: Harmonic orders to evaluate
    :param base_kvs: Dictionary of bus indices to their respective voltage levels
    :param metric: Specifies the metric to calculate ('mse', 'rmse', 'nrmse')
    :param use_pu: Use per-unit values if True
    :param vt_v: Reference target voltage level for scaling (in volt)
    :param excluded_buses: List of bus indices to exclude from calculations
    :param min_max_ref: Flag to use min-max as the reference for normalization; defaults to using mean

    :return list: Calculated error metrics for each order
    """
    if excluded_buses is None:
        excluded_buses = []
    error_metrics_per_freq = []
    num_buses = y_true.shape[1]

    for order in orders_to_evaluate:
        data_per_freq_true = y_true[:, :, order - 1]
        data_per_freq_pred = y_pred[:, :, order - 1]
        error_metrics_per_order = []

        for bus in range(num_buses):
            if bus in excluded_buses:
                error_metrics_per_order.append(None)
            else:
                voltage = base_kvs.get(bus, 400)  # Default to 400 if not specified
                scaler = calculate_scaling_factor(voltage, vt_v, use_pu)
                true_values = scaler * np.sqrt(np.sum(data_per_freq_true[:, bus] ** 2, axis=1))
                predicted_values = scaler * np.sqrt(np.sum(data_per_freq_pred[:, bus] ** 2, axis=1))
                if metric in ['mse', 'rmse']:
                    squared_errors = (true_values - predicted_values) ** 2
                    mse = np.mean(squared_errors)
                    if metric == 'mse':
                        error_metrics_per_order.append(mse)
                    else:
                        error_metrics_per_order.append(np.sqrt(mse))
                elif metric == 'nrmse':
                    ref_value = np.max(true_values) - np.min(true_values) if min_max_ref else np.mean(true_values)
                    deltas = true_values - predicted_values
                    rmse = np.sqrt(np.mean(deltas ** 2))
                    error_metrics_per_order.append(rmse / ref_value if ref_value else None)

        error_metrics_per_freq.append(error_metrics_per_order)
    return error_metrics_per_freq


def calculate_thd(data, orders_to_evaluate):
    """
    Calculate Total Harmonic Distortion (THD) for each bus.

    :param data: Measurement data (true or estimated)
    :param orders_to_evaluate: Harmonic orders to evaluate

    :return: array:
    """
    num_states = data.shape[0]
    num_buses = data.shape[1]
    thd_values = np.zeros((num_states, num_buses))

    for state in range(num_states):
        for bus in range(num_buses):
            fundamental = np.sqrt(np.sum(data[state, bus, 0] ** 2))
            harmonic_sum = sum(np.sqrt(np.sum(data[state, bus, o - 1] ** 2)) for o in orders_to_evaluate if o != 1)
            thd_values[state, bus] = 100 * harmonic_sum / fundamental if fundamental else 0
    return thd_values


def calculate_nrmse_vthd(y_true, y_pred, orders_to_evaluate, min_max_ref=False):
    """
    Calculate NRMSE for THD values between true and estimated data.

    Parameters:
    :param y_true: True THD values
    :param y_pred: Estimated THD values
    :param orders_to_evaluate: Harmonic orders to evaluate
    :param min_max_ref: Flag to use min-max as the reference; defaults to using mean
    :return: list: NRMSE values for each bus
    """
    thd_true = calculate_thd(y_true, orders_to_evaluate)
    thd_est = calculate_thd(y_pred, orders_to_evaluate)
    nrmse = []

    for bus in range(thd_true.shape[1]):
        true_values = thd_true[:, bus]
        estimated_values = thd_est[:, bus]
        ref_value = np.max(true_values) - np.min(true_values) if min_max_ref else np.mean(true_values)
        deltas = true_values - estimated_values
        rmse = np.sqrt(np.mean(np.square(deltas)))
        nrmse.append(rmse / ref_value if ref_value else None)

    return nrmse


def error_metrics_per_harmonic(y_true, y_pred, base_kvs, harmonic_orders=None, excluded_buses=None, vt_v=None):
    """
    Calculate and print MSE and RMSE for specific harmonic orders
    :param y_true: true values
    :param y_pred: predictions
    :param base_kvs: dictionary of base voltages per node
    :param harmonic_orders: harmonic orders for which to calculate the metrics
    :param excluded_buses: busses to exclude
    :param vt_v: target base voltage level to reference to
    :return:
    """
    # nmrse_thd = calculate_nrmse_vthd(data['y_test'], data['y_pred'], orders_to_evaluate_thd)
    if harmonic_orders is None:
        harmonic_orders = [1, 3, 5, 7]
    if vt_v is None:
        vt_v = 4160 / math.sqrt(3)
    mse_results = calculate_error_metric(y_true, y_pred, harmonic_orders, base_kvs, 'mse', use_pu=False,
                                         vt_v=vt_v, excluded_buses=excluded_buses, min_max_ref=False)
    for order in range(len(harmonic_orders)):
        print(f'maximum RMSE/MSE on order {harmonic_orders[order]}: {mse_results[order]}')

    nrmse_results = calculate_error_metric(y_true, y_pred, harmonic_orders, base_kvs, 'nrmse', use_pu=False,
                                           vt_v=vt_v, excluded_buses=excluded_buses, min_max_ref=False)
    for order in range(len(harmonic_orders)):
        print(f'maximum nRMSE on order {harmonic_orders[order]}: {max(nrmse_results[order])}')
