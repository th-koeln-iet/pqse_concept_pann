import os
import pickle
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model


def validate_hyperparams(hyperparams):
    if not isinstance(hyperparams['epochs'], int) or hyperparams['epochs'] < 0:
        raise ValueError("epochs must be a positive integer")

    if not isinstance(hyperparams['batch_size'], int) or hyperparams['batch_size'] <= 0:
        raise ValueError("batch_size must be a positive integer")

    if not isinstance(hyperparams['num_hidden_layers'], int) or hyperparams['num_hidden_layers'] <= 0:
        raise ValueError("num_hidden_layers must be a positive integer")

    if not isinstance(hyperparams['layer_scaling_factor'], (int, np.int64)) or hyperparams['layer_scaling_factor'] < 1:
        raise ValueError("layer_scaling_factor must be a positive non-zero integer")

    if not isinstance(hyperparams['dropout'], (float, int)) or not 0 <= hyperparams['dropout'] < 1:
        raise ValueError("dropout must be a positive number between 0 and 1")

    if not isinstance(hyperparams['learning_rate'], (int, float)) or hyperparams['learning_rate'] <= 0:
        raise ValueError("learning_rate must be a positive number")

    if not isinstance(hyperparams['skip_connections'], bool):
        raise ValueError("skip_connections must be a boolean")

    if not isinstance(hyperparams['gaussian_noise'], (int, float)) or hyperparams['gaussian_noise'] < 0:
        raise ValueError("gaussian_noise must be a positive number")

    if not isinstance(hyperparams['batch_normalization'], bool):
        raise ValueError("batch_normalization must be a boolean")


def validate_network_data(network_data):
    if not isinstance(network_data['Y_matrices_per_frequency'], np.ndarray):
        raise ValueError("Y_matrices_per_frequency must be a numpy array")

    if not isinstance(network_data['output_dims'], tuple):
        raise ValueError("output_dims must be a tuple")

    if not isinstance(network_data['input_dims'], tuple):
        raise ValueError("input_dims must be a tuple")

    for key in ['x_train', 'x_test', 'x_val', 'y_train', 'y_test', 'y_val', 'x_train_n', 'x_test_n', 'x_val_n',
                'y_train_n', 'y_test_n', 'y_val_n']:
        if key not in network_data:
            raise ValueError(f"{key} must be in network_data")
        if not isinstance(network_data[key], np.ndarray):
            raise ValueError(f"{key} must be a numpy array")

    if not isinstance(network_data['data_format'], str):
        raise ValueError("data_format must be a string")
    else:
        if network_data['data_format'] not in ['channels_first', 'channels_last']:
            raise ValueError("data_format must be either 'channels_first' or 'channels_last'")


class DNN:
    """
    Base Dense Neural Network (Keras)
    """

    def __init__(self, hyperparams, network_data, callbacks=None):
        self.callbacks = callbacks
        validate_hyperparams(hyperparams)
        validate_network_data(network_data)
        self.epochs = hyperparams['epochs']
        self.batch_size = hyperparams['batch_size']
        self.amount_of_hidden_layers = hyperparams['num_hidden_layers']
        self.layer_scaling_factor = hyperparams['layer_scaling_factor']
        self.loss_function = hyperparams['loss_function']
        self.dropout = hyperparams['dropout']
        self.activation = hyperparams['activation']
        self.learning_rate = hyperparams['learning_rate']
        self.optimizer = hyperparams['optimizer']
        self.skip_connections = hyperparams['skip_connections']
        self.gaussian_noise = hyperparams['gaussian_noise']
        self.batch_normalization = hyperparams['batch_normalization']

        self.scaler = network_data['scaler']  # output scaler, required for predictions
        self.Y_matrices_per_frequency = network_data['Y_matrices_per_frequency']
        self.frequencies = network_data['frequencies']
        self.mask_indices = network_data['mask_x_test']

        self.output_dims = network_data['output_dims']
        self.input_dims = network_data['input_dims']

        self.x_train = network_data['x_train']
        self.x_test = network_data['x_test']
        self.x_val = network_data['x_val']
        self.y_train = network_data['y_train']
        self.y_test = network_data['y_test']
        self.y_val = network_data['y_val']
        self.x_train_n = network_data['x_train_n']
        self.x_test_n = network_data['x_test_n']
        self.x_val_n = network_data['x_val_n']
        self.y_train_n = network_data['y_train_n']
        self.y_test_n = network_data['y_test_n']
        self.y_val_n = network_data['y_val_n']

        self.data_format = network_data['data_format']

        # figure out feature axes and node axis
        self.node_axis = 1 if self.data_format == 'channels_first' else -1
        self.feature_axes = [i for i in range(len(self.input_dims)) if i != self.node_axis and i != 0]

        self.model = None

    def create_layers(self):
        inp = kl.Input(shape=self.input_dims)
        # channels first means that input is of shape (batch, channels, ...)
        # frequencies is the closest to channels (see image processing RGB channels), therefor channels_first
        x = kl.Flatten(data_format=self.data_format)(inp)
        if self.gaussian_noise > 0:
            x = kl.GaussianNoise(self.gaussian_noise)(x)

        if self.batch_normalization:
            x = kl.Dense(np.prod(self.y_train_n.shape[1:]) * self.layer_scaling_factor)(x)
            x = kl.BatchNormalization()(x)
            x = kl.Activation(self.activation)(x)
        else:
            x = kl.Dense(np.prod(self.y_train_n.shape[1:]) * self.layer_scaling_factor,
                         activation=self.activation)(x)

        skip_x = x  # Store the first hidden layer's output to add skip connection later
        for i in range(0, self.amount_of_hidden_layers - 1):
            if self.batch_normalization:
                x = kl.Dense(np.prod(self.y_train_n.shape[1:]) * self.layer_scaling_factor)(x)
                x = kl.BatchNormalization()(x)
                x = kl.Activation(self.activation)(x)
            else:
                x = kl.Dense(np.prod(self.y_train_n.shape[1:]) * self.layer_scaling_factor,
                             activation=self.activation)(x)
            if self.skip_connections and i % 2 == 1:  # Adding a skip connection every two layers
                x = kl.Add()([x, skip_x])
                skip_x = x  # Update skip_x to current x for the next potential skip connection
            if self.dropout > 0:
                x = kl.Dropout(self.dropout)(x)
        # Last layer should have dynamic amount of neurons based on output_dims
        output_neurons = np.prod(self.output_dims)
        out = kl.Dense(output_neurons, activation="linear", name="prior")(x)
        out = kl.Reshape(self.output_dims)(out)
        return inp, out

    def create_model(self):
        tf.keras.backend.clear_session()
        inp, out = self.create_layers()
        model = Model(inp, out)
        if self.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:  # add more optimizers as required
            raise ValueError("Invalid optimizer type")
        model.compile(optimizer=optimizer,
                      loss=self.loss_function)
        return model

    def train(self, model, output_path, load_weights=False):
        """
        Train and evaluate the model
        :param model:
        :param output_path: path to the output directory
        :param load_weights: boolean if the weights should be loaded from the output directory
        :return: validation loss
        """
        # Create Output directory if it does not exist yet
        os.makedirs(output_path, exist_ok=True)

        model.summary()

        # Save summary
        with open(os.path.join(output_path, 'modelsummary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        checkpoint_path = os.path.join(output_path, "weights", "cp.ckpt")

        # If load_weights is True, load the latest weights and the training history
        if load_weights:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            if latest:
                model.load_weights(latest)
                print(f"Loaded weights from {latest}")
            history_path = os.path.join(output_path, 'trainHistoryDict')
            if os.path.exists(history_path):
                with open(history_path, 'rb') as file_pi:
                    old_history = pickle.load(file_pi)
            else:
                print("No history file found. Creating a new history dictionary.")
                old_history = {'loss': [], 'val_loss': []}
        else:
            old_history = {'loss': [], 'val_loss': []}

        history = model.fit(x=self.x_train_n,
                            y=self.y_train_n,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(self.x_val_n, self.y_val_n),
                            callbacks=self.callbacks,
                            verbose=2)

        # Extend old history with the new history
        if 'loss' in history.history:
            old_history['loss'].extend(history.history['loss'])
        if 'val_loss' in history.history:
            old_history['val_loss'].extend(history.history['val_loss'])

        # Save training history
        with open(os.path.join(output_path, 'trainHistoryDict'), 'wb') as file_pi:
            pickle.dump(old_history, file_pi)

        # Load model from weights to get the best model
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest:
            model.load_weights(latest)
        return model, old_history

    def predict(self, model, batch_size=None):
        """
        Evaluate model performance on given test set

        :param model: model
        :param batch_size: [optional]
        :return:
        """
        start = time.process_time()
        y_pred_n = model.predict(self.x_test_n, batch_size)
        print(f"CPU Time for prediction of test set:{time.process_time() - start}")
        # Residual = actual - predicted
        res_n = self.y_test_n - y_pred_n
        print(f"MSE on test set:{(res_n ** 2).mean()}")

        # revert normalization
        y_pred = self.scaler.revert(y_pred_n)
        return y_pred, y_pred_n
