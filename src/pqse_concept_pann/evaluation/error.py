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


def calculate_nrmse_per_frequency(data_true, data_hse, orders_to_evaluate, min_max_ref=False):
    list_return_nrmse_per_freq = []
    for order in orders_to_evaluate:
        data_per_freq_true = data_true[:, order-1, :, :]
        data_per_freq_est = data_hse[:, order-1, :, :]
        # select bus by bus
        list_return_nrmse_per_bus = []
        for bus in range(0, 44):
            if (bus == 0 or bus == 1 or bus == 20 or bus == 23 or bus == 21 or bus == 22) and order % 3 == 0:
                # 3rd order harmonics do not propagate into medium voltage level due to DYN-Transformers
                list_return_nrmse_per_bus.append(0)
            elif bus != 6 and bus != 22 and bus != 28: # do not evaluate measured nodes
                all_bus_values_true = data_per_freq_true[:, :, bus]
                all_bus_values_est = data_per_freq_est[:, :, bus]
                all_bus_values_true = [math.sqrt(d[0]**2 + d[1]**2) for d in all_bus_values_true]
                all_bus_values_est = [math.sqrt(d[0]**2 + d[1]**2) for d in all_bus_values_est]
                ref_on = 0
                if min_max_ref:
                    ref_on = np.max(all_bus_values_true) - np.min(all_bus_values_true)
                else:
                    ref_on = np.mean(all_bus_values_true)
                deltas = [all_bus_values_true[i] - all_bus_values_est[i] for i in range(len(all_bus_values_true))]
                rmse = np.sqrt(np.mean(np.square(deltas)))
                list_return_nrmse_per_bus.append(rmse/ref_on)
        list_return_nrmse_per_freq.append(list_return_nrmse_per_bus)
    return list_return_nrmse_per_freq

def calculate_mse_rmse_per_frequency(data_true, data_hse, orders_to_evaluate,
                                     use_mse=True, use_pu=False, other_ref_v:float = 1):
    """
    Calculate the MSE or RMSE per frequency for each order
    :param data_true: True values
    :param data_hse: Harmonic state estimation results
    :param orders_to_evaluate: harmonic orders to evaluate
    :param use_mse: if True, calculate MSE, otherwise RMSE
    :param use_pu: use per-unit values
    :param other_ref_v:
    :return:
    """
    list_return_nrmse_per_freq = []
    base_scale_lv = other_ref_v*1/(400/math.sqrt(3)) if use_pu else 1
    base_scale_mv = other_ref_v*1/(20000/math.sqrt(3)) if use_pu else 0.4/20
    for order in orders_to_evaluate:
        data_per_freq_true = data_true[:, order-1, :, :]
        data_per_freq_est = data_hse[:, order-1, :, :]
        # select bus by bus
        list_return_nrmse_per_bus = []
        deltas_per_order = []
        for bus in range(0, 44):
            if bus != 22 and bus != 28 and bus != 6:
                scaler = base_scale_lv
                if bus == 0 or bus == 20 or bus == 23 or bus == 1:
                    scaler = base_scale_mv
                all_bus_values_true = data_per_freq_true[:, :, bus]
                all_bus_values_est = data_per_freq_est[:, :, bus]
                all_bus_values_true = [scaler * math.sqrt(d[0]**2 + d[1]**2) for d in all_bus_values_true]
                all_bus_values_est = [scaler * math.sqrt(d[0]**2 + d[1]**2) for d in all_bus_values_est]
                delta_per_bus = [all_bus_values_true[i] - all_bus_values_est[i] for i in range(len(all_bus_values_true))]
                deltas_per_order.extend(delta_per_bus)
                # print('Bus ' + str(bus) + ': ' + str(np.mean(np.square(delta_per_bus))))
        squared_errors = np.square(deltas_per_order)
        mse = np.mean(squared_errors)
        add_to_list = mse if use_mse else np.sqrt(mse)
        list_return_nrmse_per_freq.append(add_to_list)
    return list_return_nrmse_per_freq


def calculate_nrmse_vthd(data_true, data_hse, orders_to_evaluate, min_max_ref=False):
    thd_per_bus_true, thd_per_bus_est = calculate_thd(data_true, data_hse, orders_to_evaluate)
    # select bus by bus
    list_return_nrmse_per_bus = []
    for bus in range(0, 44):
        all_bus_values_true = thd_per_bus_true[:, bus]
        all_bus_values_est = thd_per_bus_est[:, bus]
        ref_on = 0
        if min_max_ref:
            ref_on = np.max(all_bus_values_true) - np.min(all_bus_values_true)
        else:
            ref_on = np.mean(all_bus_values_true)

        deltas = [all_bus_values_true[i] - all_bus_values_est[i] for i in range(len(all_bus_values_true))]
        rmse = np.sqrt(np.mean(np.square(deltas)))
        list_return_nrmse_per_bus.append(rmse / ref_on)

    return list_return_nrmse_per_bus

def calculate_thd(data_true, data_hse, orders_to_evaluate):
    """
    Calculate the Total Harmonic Distortion (THD) for each bus
    :param data_true: True values
    :param data_hse: Harmonic state estimation results
    :param orders_to_evaluate: Harmonic orders to evaluate
    :return:
    """
    thd_per_bus_true = np.zeros((len(data_true), len(data_true[0, 0, 0])))
    thd_per_bus_est = np.zeros((len(data_true), len(data_true[0, 0, 0])))
    # calculte thd for every state and and bus
    for state in range(len(data_true)):

        for bus in range(0, 44):
            thd_denom_u_sqared_true = 0
            thd_denom_u_sqared_est = 0
            fundamental_squared_true = math.sqrt(
                data_true[state, 0, 0, bus] ** 2 + data_true[state, 0, 1, bus] ** 2)
            fundamental_squared_est = math.sqrt(data_hse[state, 0, 0, bus] ** 2 + data_hse[state, 0, 1, bus] ** 2)
            for o in orders_to_evaluate:
                val_true = math.sqrt(data_true[state, o - 1, 0, bus] ** 2 + data_true[state, o - 1, 1, bus] ** 2)
                thd_denom_u_sqared_true += (val_true ** 2)
                val_est = math.sqrt(data_hse[state, o - 1, 0, bus] ** 2 + data_hse[state, o - 1, 1, bus] ** 2)
                thd_denom_u_sqared_est += (val_est ** 2)
            thd_per_bus_true[state, bus] = 100 * math.sqrt(thd_denom_u_sqared_true) / fundamental_squared_true
            thd_per_bus_est[state, bus] = 100 * math.sqrt(thd_denom_u_sqared_est) / fundamental_squared_est
    return thd_per_bus_true, thd_per_bus_est
