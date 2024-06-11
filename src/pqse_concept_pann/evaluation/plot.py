import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pqse_concept_pann.evaluation.error import get_max_error_split_axis, get_mae_split_axis
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
        fig.savefig(os.path.join(save_path, f"{mode}.png"))
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
        fig.savefig(os.path.join(save_path, f"train_val_loss.png"))
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
            plt.plot(sorted_losses, cdf, '--', color=color, label=model_name)
            specific_string_found = True
        else:
            plt.plot(sorted_losses, cdf, color=color, label=model_name)

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
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'cdf_{label}.png'))
    if show_plot:
        plt.show()
    plt.close()


def plot_estimations_wrapper(predictions, grid_name, y_test, x_test, mask_x_test, mask_y_test=None, iteration=1,
                             harmonic=1, save_path=None, show_plot=True):
    """

    :param predictions: dictionary of key <Model identifier> and value <predictions dict>
    :param grid_name: name of the grid
    :param y_test: in cartesian form
    :param x_test: in cartesian form
    :param mask_x_test: mask indices of x_test
    :param mask_y_test: mask indices of y_test, default None = all buses
    :param iteration: which iteration to plot
    :param harmonic: which harmonic to plot
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
                     show_plot=show_plot)


def plot_estimations(y_test_polar, x_test_polar, y_pred_polar, title, mask_x_test, mask_y_test=None, save_path=None,
                     show_plot=True):
    """
    Plot the true values, the predictions and the noiseful input for each bus.
    :param y_test_polar: true values
    :param x_test_polar: noiseful input
    :param y_pred_polar: dictionary of model name -> predictions
    :param title: Plot title
    :param mask_x_test: mask indices of x_test (i.e. measurement locations)
    :param mask_y_test: mask indices for y_test (i.e. true values), default None = all buses
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
    markers = ['s', '^', 'D']  # Additional markers can be added as needed
    edgecolors = ['g', 'orange', 'brown']
    for idx, (model_name, y_pred) in enumerate(y_pred_polar.items()):
        plt.scatter(bus_numbers, y_pred, label=f'Prediction ({model_name})', edgecolors=edgecolors[idx],
                    marker=markers[idx % len(markers)], facecolors='none')

    # Plotting x_test considering indices with outlined symbols
    if mask_x_test is not None:
        plt.scatter(mask_x_test, x_test_polar, label='Noiseful input', facecolors='b', marker='x')

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
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    plt.close()


def evaluate(predictions, data, save_path=None, show_plot=True):
    """
    Evaluate predictions
    :param predictions: dictionary of model predictions with model label as key and prediction data as value
    :param data: dictionary of network data
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
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1000, harmonic=1, save_path=save_path, show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1000, harmonic=5, save_path=save_path, show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1000, harmonic=7, save_path=save_path, show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1000, harmonic=9, save_path=save_path, show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1000, harmonic=13, save_path=save_path, show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1000, harmonic=15, save_path=save_path, show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1000, harmonic=17, save_path=save_path, show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1600, harmonic=1, save_path=save_path, show_plot=show_plot)
    plot_estimations_wrapper(predictions, data['grid_name'], data['y_test'], data['x_test'], data['mask_x_test'],
                             data['mask_y_test'],
                             iteration=1600, harmonic=7, save_path=save_path, show_plot=show_plot)


def plot_heatmaps(predictions, data, complex_axis=3, save_path=None, show_plot=True, magnitude_only=False):
    # plots for each model:
    for key, value in predictions.items():
        # convert to polar
        y_pred_polar = cartesian_to_polar(value['y_pred'], axis=complex_axis)
        y_test_polar = cartesian_to_polar(data['y_test'], axis=complex_axis)
        x_test_polar = cartesian_to_polar(data['x_test'], axis=complex_axis)
        res = y_test_polar - y_pred_polar
        df_titles = ['Magnitude', 'Angle']
        # MAE plots
        dfs = get_mae_split_axis(y_test_polar, y_pred_polar, data['frequencies'], feature_axis=complex_axis,
                                 error_axis=0)
        if magnitude_only:
            dfs = (dfs[0],)
        plot_heatmap(list(dfs), df_titles,
                     title=f"{key} MAE",
                     save_path=save_path, show_plot=show_plot)
        plot_heatmap(list(dfs), df_titles,
                     title=f"{key} MAE", drop50hz=True,
                     save_path=save_path, show_plot=show_plot)

        df_titles = ['Real', 'Imaginary']
        # Max error plots
        res_n = data['y_test_n'] - value['y_pred_n']
        dfs = get_max_error_split_axis(data['y_test_n'], value['y_pred_n'], data['frequencies'],
                                       feature_axis=complex_axis,
                                       error_axis=0)
        if magnitude_only:
            dfs = (dfs[0],)
        plot_heatmap(dfs, df_titles,
                     title=f"{key} Max Error (Normalized)",
                     save_path=save_path, show_plot=show_plot)
        dfs = get_max_error_split_axis(data['y_test'], value['y_pred'], data['frequencies'], feature_axis=complex_axis,
                                       error_axis=0)
        if magnitude_only:
            dfs = (dfs[0],)
        plot_heatmap(dfs, df_titles,
                     title=f"{key} Max Error",
                     save_path=save_path, show_plot=show_plot)
        plot_heatmap(dfs, df_titles,
                     title=f"{key} Max Error", drop50hz=True,
                     save_path=save_path, show_plot=show_plot)


def plot_heatmap(dfs, df_titles, title="", drop50hz=False, save_path=None, show_plot=True):
    """
    Plot dataframes of shape (batch, frequencies, nodes) as heatmaps
    :param dfs: list of pandas dataframes
    :param df_titles: list of strings, titles for each dataframe
    :param title:
    :param drop50hz: if True 50Hz is dropped
    :param save_path: if provided, plot will be saved at specified location
    :param show_plot: if True plot will be shown
    :return:
    """
    fig, axs = plt.subplots(nrows=len(dfs), figsize=(11.7, 8.27))
    for i, df in enumerate(dfs):
        if drop50hz:
            df = df.drop(50, axis=1)
        if len(dfs) == 1:
            s = sns.heatmap(df, ax=axs)
        else:
            s = sns.heatmap(df, ax=axs[i])
        s.set(xlabel='Frequency', ylabel='Node', title=df_titles[i])
    fig.suptitle(title)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if drop50hz:
            save_path = os.path.join(save_path, title + "_drop50_" + '.png')
        else:
            save_path = os.path.join(save_path, title + '.png')
        fig.savefig(save_path)
    if show_plot:
        plt.tight_layout()
        plt.show()
    plt.close()