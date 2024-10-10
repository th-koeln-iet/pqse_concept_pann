import math
import os
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pqse_concept_pann.evaluation.error import get_max_error_split_axis, get_mae_split_axis, calculate_thd
from pqse_concept_pann.tools.split_complex import cartesian_to_polar


def plot_losses(losses, save_path=None, show_plot=True):
    if losses is not None and len(losses) > 0:
        # plot loss history
        val_loss = pd.DataFrame({key: val['val_loss'] for key, val in losses.items()})
        train_loss = pd.DataFrame({key: val['loss'] for key, val in losses.items()})
        plot_history(val_loss, mode='val_loss', save_path=save_path, show_plot=show_plot)
        plot_history(train_loss, mode='loss', save_path=save_path, show_plot=show_plot)
        plot_train_val_loss_history(train_loss, val_loss, save_path=save_path, show_plot=show_plot)
        print(f"Minimum val loss:{val_loss.min()}")


def plot_history(data, mode='val_loss', save_path=None, show_plot=True):
    """
    Plot the loss history of a model
    :param data: loss history
    :param mode: 'val_loss' or 'loss'
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    df = pd.DataFrame(data=data)
    ax = df.plot(figsize=(11.7, 8.27), logx=True, logy=True)
    if mode == 'val_loss':
        ax.set_title('Validation Loss over Epochs')
    elif mode == 'loss':
        ax.set_title('Training Loss over Epochs')
    else:
        raise ValueError("mode must be either 'val_loss' or 'loss'")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    fig = ax.get_figure()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"{mode}.png"), dpi=300)
        fig.savefig(os.path.join(save_path, f"{mode}.svg"), format='svg')
    if show_plot:
        plt.show()
    plt.close()


def plot_train_val_loss_history(df_train, df_val, save_path=None, show_plot=True):
    """
    Plot the training and validation loss history of models
    :param df_val: validation loss history as pandas dataframe with key model name
    :param df_train: training loss history as pandas dataframe with key model name
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """

    ax = None
    colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, len(df_train.columns)))

    for col, color in zip(df_train.columns, colors):
        ax = df_train[col].plot(figsize=(11.7, 8.27), logx=False, logy=True, ax=ax, color=color, label=col + '_train',
                                linestyle='-')
        df_val[col].plot(logx=False, logy=True, ax=ax, color=color, label=col + '_val', linestyle='--')

    # Add title, labels, and legend
    ax.set_title('Training and Validation Loss over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (Log Scale)')
    ax.legend()

    fig = ax.get_figure()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"train_val_loss.png"), dpi=300)
        fig.savefig(os.path.join(save_path, f"train_val_loss.svg"), format='svg')
    if show_plot:
        plt.show()
    plt.close()


def plot_cdf(prediction_losses, label='MAPE', dashed_line_str='input noise', color_str='Gauss Layer 0.02',
             save_path=None, show_plot=True):
    """
    Plot the CDF of the <label> for each model.
    :param prediction_losses: dictionary of model name -> list of flattened MAPEs, MAEs, ...
    :param label: MAPE, MAE or similar
    :param dashed_line_str: optional parameter that results in dashed lines if in model name
    :param color_str: optional parameter for different color if in model name
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return: None
    """
    # Flag to check if the specific string is encountered at least once
    specific_string_found = False

    for model_name, losses in prediction_losses.items():
        # Sort the losses
        sorted_losses = np.sort(losses)

        # Generate the CDF
        cdf = np.arange(1, len(sorted_losses) + 1) / float(len(sorted_losses))
        if color_str == "":
            color = None
        else:
            if color_str in model_name:
                color = 'r'
            else:
                color = 'b'
        # Check for the specific string and plot accordingly
        if dashed_line_str in model_name:
            plt.plot(sorted_losses, cdf, '--', label=model_name)
            specific_string_found = True
        else:
            plt.plot(sorted_losses, cdf, label=model_name)

    # Customize the graph
    plt.title(f'CDF of {label} for Different Models')

    if label == "MAE":
        plt.xlabel(f'Mean Absolute Error ({label})')
    elif label == "MAPE":
        plt.xlabel(f'Mean Absolute Percentage Error ({label})')
    else:
        plt.xlabel(f'{label}')

    plt.ylabel('Cumulative Distribution Function (CDF)')

    # Add the legend for the specific string if it is found
    if specific_string_found:
        plt.legend(title=f'Dashed Line = {dashed_line_str}')
    else:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'cdf_{label}.png'), dpi=300)
        plt.savefig(os.path.join(save_path, f'cdf_{label}.svg'), format='svg')
    if show_plot:
        plt.show()
    plt.close()


def plot_estimations_wrapper(predictions, grid_name, y_test, x_test, mask_x_test, mask_y_test=None, iteration=1,
                             input_noise=True, harmonic=1, save_path=None, show_plot=True):
    """

    :param predictions: dictionary of key <Model identifier> and value <predictions dict>
    :param grid_name: name of the grid
    :param y_test: in cartesian form
    :param x_test: in cartesian form
    :param mask_x_test: mask indices of x_test
    :param mask_y_test: mask indices of y_test, default None = all buses
    :param iteration: which iteration to plot
    :param harmonic: which harmonic to plot
    :param input_noise: if measurement noise is added to the input, set to True to get accurate labels in plots
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    y_preds = {}
    for key, value in predictions.items():
        y_preds[key] = cartesian_to_polar(value['y_pred'], axis=3)[iteration, :, harmonic - 1, 0]
    y_test = cartesian_to_polar(y_test, axis=3)
    y_test_single_it = y_test[iteration, :, harmonic - 1, 0]
    x_test = cartesian_to_polar(x_test, axis=3)
    x_test_single_it = x_test[iteration, :, harmonic - 1, 0]
    title = f"Estimation harmonic {harmonic} for {grid_name} at iteration {iteration}"
    if save_path is not None:
        path = os.path.join(save_path, f'estimations_it{iteration}_h{harmonic}.png')
        os.makedirs(save_path, exist_ok=True)
    else:
        path = None
    plot_estimations(y_test_single_it, x_test_single_it, y_preds, title, mask_x_test, mask_y_test, save_path=path,
                     show_plot=show_plot, input_noise=input_noise)


def plot_estimations(y_test_polar, x_test_polar, y_pred_polar, title, mask_x_test, mask_y_test=None, input_noise=True,
                     save_path=None, show_plot=True):
    """
    Plot the true values, the predictions and the noiseful input for each bus.
    :param y_test_polar: true values
    :param x_test_polar: noiseful input
    :param y_pred_polar: dictionary of model name -> predictions
    :param title: Plot title
    :param mask_x_test: mask indices of x_test (i.e. measurement locations)
    :param mask_y_test: mask indices for y_test (i.e. true values), default None = all buses
    :param input_noise: if True, noise is added to the input, then the label is changed in this method
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    bus_numbers = list(range(0, len(next(iter(y_pred_polar.values())))))  # somewhat hacky way to get all nodes
    if mask_y_test is None:
        mask_y_test = bus_numbers
    plt.figure(figsize=(10, 6))

    # Plotting y_test with outlined symbols
    plt.scatter(mask_y_test, y_test_polar, label='True value', edgecolors='r', marker='o', facecolors='none')

    # Plotting y_pred_polar considering multiple models with outlined symbols
    markers = ['s', '^', 'D', 'h', '2']  # Additional markers can be added as needed
    edgecolors = ['g', 'orange', 'brown', 'purple', 'darkred']
    for idx, (model_name, y_pred) in enumerate(y_pred_polar.items()):
        plt.scatter(bus_numbers, y_pred, label=f'Prediction ({model_name})', edgecolors=edgecolors[idx],
                    marker=markers[idx % len(markers)], facecolors='none')

    # Plotting x_test considering indices with outlined symbols
    if mask_x_test is not None:
        label = 'Noiseful Input' if input_noise else 'Input'
        plt.scatter(mask_x_test, x_test_polar, label=label, facecolors='b', marker='x')

    # Setting x-axis and y-axis labels
    plt.xlabel('Bus Number')
    plt.ylabel('Voltage Magnitude (V)')
    plt.title(title)
    # Setting y-axis limits to show only relevant parts
    all_values = np.concatenate(
        [y_test_polar, x_test_polar if mask_x_test is not None else []] + list(y_pred_polar.values()))
    lower_limit = np.min(all_values) - 0.1 * (np.max(all_values) - np.min(all_values))
    upper_limit = np.max(all_values) + 0.1 * (np.max(all_values) - np.min(all_values))
    plt.ylim([lower_limit, upper_limit])

    # Adding a legend
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path.replace('png', 'svg'), format='svg')
    if show_plot:
        plt.show()
    plt.close()


def evaluate(predictions, data, input_noise=True, save_path=None, show_plot=True):
    """
    Evaluate predictions
    :param predictions: dictionary of model predictions with model label as key and prediction data as value
    :param data: dictionary of network data
    :param input_noise: if measurement noise is added to the input, set to True to get accurate labels in plots
    :param save_path: path to save plots to, default None
    :param show_plot: True if plots shall be shown
    :return:
    """
    complex_axis = 3
    # convert to polar
    y_preds = {}
    for key, value in predictions.items():
        y_preds[key] = cartesian_to_polar(value['y_pred'], axis=complex_axis)
    y_test = cartesian_to_polar(data['y_test'], axis=complex_axis)
    ## plot estimations for any iteration here
    ## to get the time of the day of a given iteration (in 24h format) calculate: (iteration % (4*24))/4
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1026, harmonic=1, input_noise=input_noise, save_path=save_path,
                             show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1026, harmonic=7, input_noise=input_noise, save_path=save_path,
                             show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1026, harmonic=9, input_noise=input_noise, save_path=save_path,
                             show_plot=show_plot)


def plot_heatmaps(predictions, data, complex_axis=3, save_path=None, show_plot=True, magnitude_only=False):
    # plots for each model:
    for key, value in predictions.items():
        # convert to polar
        y_pred_polar = cartesian_to_polar(value['y_pred'], axis=complex_axis)
        y_test_polar = cartesian_to_polar(data['y_test'], axis=complex_axis)
        x_test_polar = cartesian_to_polar(data['x_test'], axis=complex_axis)
        res = y_test_polar - y_pred_polar
        df_titles = ['Mean Absolute Error of Magnitude [V]', 'Mean Absolute Error of Angle']
        # MAE plots
        dfs = get_mae_split_axis(y_test_polar, y_pred_polar, data['frequencies'], feature_axis=complex_axis,
                                 error_axis=0)
        if magnitude_only:
            dfs = (dfs[0],)
        plot_heatmap(list(dfs), df_titles,
                     title=f"{key} MAE",
                     save_path=save_path, show_plot=show_plot)
        plot_heatmap(list(dfs), df_titles,
                     title=f"{key} MAE", drop_fundamental=True,
                     save_path=save_path, show_plot=show_plot)
        plot_heatmap_subplots(dfs[0], df_titles=df_titles, title=f"{key} MAE", save_path=save_path, show_plot=show_plot)
        # Max error plots
        dfs = get_max_error_split_axis(y_test_polar, y_pred_polar, data['frequencies'], feature_axis=complex_axis,
                                       error_axis=0)
        df_titles = ['Maximum Absolute Error of Magnitude [V]', 'Maximum Absolute Error of Angle']
        if magnitude_only:
            dfs = (dfs[0],)
        plot_heatmap(dfs, df_titles,
                     title=f"{key} Max Error",
                     save_path=save_path, show_plot=show_plot)
        plot_heatmap_subplots(dfs[0], df_titles=df_titles, title=f"{key} Max Error", save_path=save_path, show_plot=show_plot)
        plot_heatmap(dfs, df_titles,
                     title=f"{key} Max Error", drop_fundamental=True,
                     save_path=save_path, show_plot=show_plot)


def plot_heatmap(dfs, df_titles, title="", drop_fundamental=False, save_path=None, show_plot=True):
    """
    Plot dataframes of shape (batch, frequencies, nodes) as heatmaps
    :param dfs: list of pandas dataframes
    :param df_titles: list of strings, titles for each dataframe
    :param title:
    :param drop_fundamental: if True fundamental frequency is dropped
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(nrows=len(dfs), figsize=(11.7, 8.27))
    for i, df in enumerate(dfs):
        if drop_fundamental:
            column_max_val = df.max().idxmax()
            if np.isclose(float(column_max_val), 50, rtol=1e-3) or np.isclose(float(column_max_val), 60, rtol=1e-3):
                df = df.drop(column_max_val, axis=1)
        if len(dfs) == 1:
            s = sns.heatmap(df, ax=axs)
        else:
            s = sns.heatmap(df, ax=axs[i])
        s.set(xlabel='Frequency [Hz]', ylabel='Node', title=df_titles[i])
    # fig.suptitle(title)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if drop_fundamental:
            save_path = os.path.join(save_path, title + "_drop-fund_")
        else:
            save_path = os.path.join(save_path, title)
        fig.savefig(save_path + '.png', dpi=300)
        fig.savefig(save_path + '.svg', format='svg')
    if show_plot:
        plt.show()
    plt.close()


def plot_heatmap_subplots(df, df_titles, title="", save_path=None, show_plot=True, n_major_intervals=5):
    """
    Plot fundamental and harmonics heatmaps on two scales in one plot
    :param df: pandas dataframe with the dataset
    :param df_titles: list of strings, titles for each dataframe
    :param title: model name or similar
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :param n_major_intervals: how many intervals shall the new colorbar have
    :return:
    """

    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(ncols=3, figsize=(14, 9), gridspec_kw={'width_ratios': [1, 1, 19], 'wspace': 0.25})

    column_max_val = df.max().idxmax()
    df_only_fundamental = df[[column_max_val]]
    df_drop_fundamental = df.drop(column_max_val, axis=1)

    df_min = df_only_fundamental.min().min()
    df_max = df_only_fundamental.max().max()

    # Interpolate values for own colorbar
    interpolated_values = np.linspace(df_max, df_min, 1000)
    interpolated_series = np.linspace(1000, 1, 1000)
    df_scale = pd.DataFrame(interpolated_series)
    df_scale.index = interpolated_values

    # Add a heatmap as custom colorbar
    s_f_scale = sns.heatmap(df_scale, ax=axs[0], cbar=False, cmap="viridis")

    new_ticks = []
    new_tick_labels = []
    old_ticks = s_f_scale.get_yticks()
    delta_ticks = (old_ticks[-1] - old_ticks[0]) / n_major_intervals
    df_max_start = (df_max - (df_max % 5))
    delta_tick_labels = (df_max_start - (df_min - (df_min % 5))) / n_major_intervals
    for i in range(0, n_major_intervals):
        tick = old_ticks[0] + delta_ticks * i
        new_ticks.append(tick)
        label = df_max_start - delta_tick_labels*i
        new_tick_labels.append(str(int(math.trunc(label))))

    axs[0].set_yticks(new_ticks, new_tick_labels)
    axs[0].set_title(r'$e_{M_{F}} [V]$')
    axs[0].set_ylabel('')  # remove y-axis label
    axs[0].set_xlabel('')  # remove x-axis label
    axs[0].set_xticks([])  # remove x-ticks

    # Actual data
    s_f = sns.heatmap(df_only_fundamental, ax=axs[1], cbar=False, cmap="viridis")
    s_h = sns.heatmap(df_drop_fundamental, ax=axs[2])
    cbar = s_h.collections[0].colorbar
    cbar.ax.set_title(r'$e_{M_{H}} [V]$')

    axs[2].set_yticklabels([])
    s_f.set(ylabel='Node')
    plt.setp(s_f.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(s_f.yaxis.get_majorticklabels(), rotation=0)
    axs[1].set_title('50Hz')
    axs[2].set_title('Harmonics')
    fig.suptitle(df_titles[0])
    # plt.text(1.0, 900, 'Frequency [Hz]')  # 0.5 0.05
    s_h.set(xlabel='Frequency [Hz]')
    plt.yticks(rotation=0)
    s_h.tick_params(labelrotation=45)
    # plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, title + "_combined")
        fig.savefig(save_path + '.png', dpi=300)
        fig.savefig(save_path + '.svg', format='svg')
    if show_plot:
        plt.show()
    plt.close()


def plot_thd_bus(y_true, y_pred, orders_to_evaluate, show_busses, save_path=None, show_plot=True):
    thd_true = calculate_thd(y_true, orders_to_evaluate)
    thd_est = calculate_thd(y_pred, orders_to_evaluate)
    start_day = 4
    end_day = 5
    # Create subplots
    fig, axs = plt.subplots(len(show_busses), 1, figsize=(10, 8), sharex=True)
    time_of_day_minutes = range(4 * 24 * (start_day - 1), 4 * 24 * (end_day - 1))  # time of day from 0 to 24 hours
    start_time = datetime(2024, 1, 1)
    time_of_day = [start_time + timedelta(minutes=tm * 15) for tm in
                   time_of_day_minutes]  # time of day from 0 to 24 hours
    for i, bus in enumerate(show_busses):
        thd_data_true = thd_true[time_of_day_minutes, bus]
        thd_data_est = thd_est[time_of_day_minutes, bus]

        # Plotting the data for each subplot
        axs[i].plot(time_of_day, thd_data_true, label='True')
        axs[i].plot(time_of_day, thd_data_est, label='Est.')
        axs[i].set_ylabel('$THD_{V} [\\%]$')
        axs[i].legend()
        axs[i].set_title('Busbar ' + str(bus))

    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    # Set x-axis limits
    min_time = start_time + timedelta(minutes=(start_day - 1) * 24 * 60)  # start time
    max_time = start_time + timedelta(minutes=(end_day - 1) * 24 * 60 - 1)  # end time

    for ax in axs:
        ax.set_xlim(min_time, max_time)

    plt.xticks(rotation=45)
    # Common x-axis label
    plt.xlabel('time of day [HH:MM]')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'THD_busses.png'), dpi=300)
        plt.savefig(os.path.join(save_path, 'THD_busses.svg'), format='svg')
    if show_plot:
        plt.show()
    plt.close()
