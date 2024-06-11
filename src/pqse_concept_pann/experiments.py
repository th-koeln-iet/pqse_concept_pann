import configparser
import os
from evaluation.plot import evaluate, plot_heatmaps, plot_losses, plot_cdf
from layers import AdjacencyPrunedLayer
from tools import DimMinMaxScaler, SplitComplexMode, SplitComplexConverter, mean_absolute_error
from models import DNN
from models import PANN
from models.callbacks import MinimalLossSaveModelCheckpoint, CyclicLR
from preprocessing import read_data, add_measurement_noise
import numpy as np
import tensorflow as tf
import random

# Set random seed for reproducibility
# full reproducibility also requires setting shuffle and use_multiprocessing to False in model fit
np.random.seed(1)
tf.set_random_seed(2)
random.seed(3)

config = configparser.ConfigParser()
config.sections()
config_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
config.read(os.path.join(config_path, 'config.ini'))
data_path = os.path.abspath(os.path.join(config_path, os.path.expanduser(config['DEFAULT']['data_path'])))

# Alter the following parameters
EPOCHS = 0  # Epochs to train
USE_TRAINED_MODEL = True  # use existing weights of trained model?
# If EPOCHS is set to a number > 0 and USE_TRAINED_MODEL is set, the model will continue training on the loaded weights


def state_estimation(hyperparams: dict, network_data: dict, model_class, callbacks: list, exp_path: str, label: str,
                     custom_layer=None, load_weights: bool = False):
    """
    Create and train a model for state estimation, and return the loss and predictions
    :param hyperparams:
    :param network_data:
    :param model_class:
    :param callbacks:
    :param exp_path:
    :param label:
    :param custom_layer:
    :param load_weights:
    :return:
    """
    loss = {}
    predictions = {}
    if model_class == PANN:
        m = model_class(hyperparams, network_data, callbacks, custom_layer=custom_layer)
    else:
        m = model_class(hyperparams, network_data, callbacks)
    model = m.create_model()
    model, loss[label] = m.train(model, exp_path, load_weights=load_weights)
    y_pred, y_pred_n = m.predict(model)
    pred = {'y_pred': y_pred, 'y_pred_n': y_pred_n}
    predictions[label] = pred
    return loss, predictions


def exp_dnn_config():
    label = "DNN"
    exp_path = os.path.join(data_path, label)
    hyperparams = {
        'batch_size': 16384,
        'num_hidden_layers': 2,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-4,
        'layer_scaling_factor': 1, # set scaling factor to 2 to achieve the same amount of neurons per layer as in PANN
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.00,  # standard deviation
        'batch_normalization': False,
    }
    callbacks = [MinimalLossSaveModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'), grace_period=500, min_eps_percent=1e-3,
                                                monitor='val_loss', verbose=1),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200.)]
    model_class = DNN
    custom_layer = None
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def exp_dnn_gauss_config():
    label = "DNN Gauss 0.02"
    exp_path = os.path.join(data_path, label)
    hyperparams = {
        'batch_size': 16384,
        'num_hidden_layers': 2,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-4,
        'layer_scaling_factor': 1,
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.02,  # standard deviation
        'batch_normalization': False,
    }
    callbacks = [MinimalLossSaveModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'), grace_period=500, min_eps_percent=1e-3,
                                                monitor='val_loss', verbose=1),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200.)]
    model_class = DNN
    custom_layer = None
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def exp_pann_config():
    label = "PANN"
    exp_path = os.path.join(data_path, label)
    hyperparams = {
        'batch_size': 16384,
        'num_hidden_layers': 6,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-4,
        'layer_scaling_factor': 1,
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.00,  # standard deviation
        'batch_normalization': False,
    }
    callbacks = [MinimalLossSaveModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'), grace_period=500, min_eps_percent=1e-3,
                                                monitor='val_loss', verbose=1),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200.)]
    model_class = PANN
    custom_layer = AdjacencyPrunedLayer
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def exp_pann_gauss_config():
    label = "PANN Gauss 0.02"
    exp_path = os.path.join(data_path, label)
    hyperparams = {
        'batch_size': 16384,
        'num_hidden_layers': 6,
        'dropout': 0,
        'loss_function': 'mse',
        'activation': 'leaky_relu',
        'optimizer': 'adam',
        'learning_rate': 5e-4,
        'layer_scaling_factor': 1,
        'epochs': EPOCHS,
        'skip_connections': True,
        'gaussian_noise': 0.02,  # standard deviation
        'batch_normalization': False,
    }
    callbacks = [MinimalLossSaveModelCheckpoint(filepath=os.path.join(exp_path, 'weights', 'cp.ckpt'), grace_period=500, min_eps_percent=1e-3,
                                                monitor='val_loss', verbose=1),
                 CyclicLR(min_lr=hyperparams['learning_rate'] / 5, max_lr=hyperparams['learning_rate'] * 2,
                          step_size=200.)]
    model_class = PANN
    custom_layer = AdjacencyPrunedLayer
    return hyperparams, callbacks, model_class, custom_layer, exp_path, label


def load_data():
    # Load data
    modes = [SplitComplexMode.CARTESIAN, SplitComplexMode.EXPONENTIAL]
    network_data = {
        'grid_name': "CigreLVDist",
        'scaler': DimMinMaxScaler(),
        'test_size': 0.1,
        'validation_size': 0.2,
        'data_format': 'channels_last',
        'mask_x_test': [6, 22, 28],
        'mask_y_test': None,  # mask of target values
        'split_complex_modes': modes
    }

    # from files
    input_converter = SplitComplexConverter(modes, target_axis=3)
    target_converter = SplitComplexConverter(modes[0], target_axis=3)
    data = read_data(data_path, network_data, input_converter, target_converter, transpose_data_format=True)
    return data


def compare_experiments(configs, data, measurement_noise=0.01, save_path = None):
    """
    Compare multiple experiments to each other
    :param configs: experiment configurations
    :param data: network data
    :param measurement_noise: add measurement noise to the data (if None, no measurement noise is added)
    :param save_path: path to save the plots
    :return: predictions and losses
    """
    predictions = {}
    losses = {}
    if measurement_noise is not None:
        from copy import deepcopy
        data_updated = deepcopy(data)
        data_updated['x_test_n'] = add_measurement_noise(data_updated['x_test_n'], 0.01)
        data_updated['x_test'] = data_updated['input_scaler'].revert(
            data_updated['x_test_n'])  # apply same noise to non-scaled data as well
        load_weights = False  # no training necessary, only prediction
    else:
        data_updated = data
        load_weights = USE_TRAINED_MODEL
    for cfg in configs:
        hyperparams, callbacks, model_class, custom_layer, exp_path, label = cfg
        if measurement_noise:
            hyperparams['epochs'] = 0  # no training necessary, only prediction
        loss, preds = state_estimation(hyperparams, data_updated, model_class, callbacks, exp_path, label,
                                       custom_layer=custom_layer, load_weights=load_weights)
        losses.update(loss)
        predictions.update(preds)
    evaluate(predictions, data_updated, save_path=save_path, show_plot=False)
    # plot residual density
    mae = {}
    for key, value in predictions.items():
        mae[key] = mean_absolute_error(data['y_test_n'], value['y_pred_n'], axis=0).flatten()
    plot_cdf(mae, label='MAE', color_str='DNN', save_path=save_path, show_plot=False)
    return predictions, losses


if __name__ == '__main__':
    # Load data from pickle files
    data = load_data()

    # Run experiments

    ## Base models
    save_path = os.path.join(data_path, 'plots')
    base_model_configs = [exp_dnn_config(), exp_pann_config()]
    predictions, losses = compare_experiments(base_model_configs, data, measurement_noise=None, save_path=save_path)
    plot_heatmaps(predictions, data, complex_axis=3, save_path=save_path, show_plot=False,
                  magnitude_only=True)
    plot_losses(losses, save_path=save_path, show_plot=False)

    ### with input noise
    save_path = os.path.join(data_path, 'plots_noise')
    compare_experiments(base_model_configs, data, measurement_noise=0.01, save_path=save_path)

    ## Gaussian noise models
    save_path = os.path.join(data_path, 'plots_gauss')
    gauss_model_configs = [exp_dnn_gauss_config(), exp_pann_gauss_config()]
    predictions, losses = compare_experiments(gauss_model_configs, data, measurement_noise=None, save_path=save_path)
    plot_heatmaps(predictions, data, complex_axis=3, save_path=save_path, show_plot=False,
                  magnitude_only=True)
    plot_losses(losses, save_path=save_path, show_plot=False)

    ### with input noise
    save_path = os.path.join(data_path, 'plots_noise_gauss')
    compare_experiments(gauss_model_configs, data, measurement_noise=0.01, save_path=save_path)

