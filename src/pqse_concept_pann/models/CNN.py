import tensorflow as tf
import numpy as np
from models.DNN import DNN


class CNN(DNN):
    def __init__(self, hyperparams, network_data, callbacks=None):
        super(CNN, self).__init__(hyperparams, network_data, callbacks)
        self.num_heads = hyperparams['num_heads']
        self.num_layers = hyperparams['num_hidden_layers']
        self.dropout_rate = hyperparams['dropout']

    def create_layers(self):
        inp = tf.keras.layers.Input(shape=self.input_dims)

        x = tf.keras.layers.Conv2D(128, (1, 1), 1, padding='same', use_bias=False)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        for _ in range(self.num_layers):
            x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Flatten()(x)
        output_neurons = tf.math.reduce_prod(self.output_dims)
        out = tf.keras.layers.Dense(output_neurons, activation="linear", name="prior")(x)
        out = tf.keras.layers.Reshape(self.output_dims)(out)
        return inp, out

    def create_model(self):
        tf.keras.backend.clear_session()
        inp, out = self.create_layers()
        model = tf.keras.models.Model(inp, out)
        if self.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer type")
        model.compile(optimizer=optimizer,
                      loss=self.loss_function)
        return model


