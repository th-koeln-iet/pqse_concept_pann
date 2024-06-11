import numpy as np
import tensorflow.keras.layers as kl

from pqse_concept_pann.layers import AdmittanceWeightedLayer
from pqse_concept_pann.models import DNN


class PANN(DNN):
    """
    Physics Aware Deep Neural Network (Keras)
    """

    def __init__(self, hyperparams, network_data, callbacks, custom_layer=AdmittanceWeightedLayer):
        super().__init__(hyperparams, network_data, callbacks)
        self.custom_layer = custom_layer

    def create_layers(self):
        inp = kl.Input(shape=self.input_dims)
        if self.gaussian_noise > 0:
            x = kl.GaussianNoise(self.gaussian_noise)(inp)
        else:
            x = inp
        if self.batch_normalization:
            x = self.custom_layer(y_mats=self.Y_matrices_per_frequency,
                                  mask_indices=self.mask_indices,
                                  layer_scaling_factor=self.layer_scaling_factor, data_format=self.data_format)(x)
            x = kl.BatchNormalization(axis=self.feature_axes)(x)
            x = kl.Activation(self.activation)(x)
        else:
            x = self.custom_layer(y_mats=self.Y_matrices_per_frequency,
                                  mask_indices=self.mask_indices, activation="leaky_relu",
                                  layer_scaling_factor=self.layer_scaling_factor, data_format=self.data_format)(x)
        skip_x = x  # Store the first hidden layer's output to add skip connection later

        for i in range(0, self.amount_of_hidden_layers - 1):
            if self.batch_normalization:
                x = self.custom_layer(y_mats=self.Y_matrices_per_frequency,
                                      mask_indices=None,
                                      layer_scaling_factor=self.layer_scaling_factor, data_format=self.data_format)(x)
                x = kl.BatchNormalization(axis=self.feature_axes)(x)
                x = kl.Activation(self.activation)(x)
            else:
                x = self.custom_layer(y_mats=self.Y_matrices_per_frequency,
                                      mask_indices=None, activation="leaky_relu",
                                      layer_scaling_factor=self.layer_scaling_factor, data_format=self.data_format)(x)
            if self.skip_connections and i % 2 == 1:  # Adding a skip connection every two layers
                x = kl.Add()([x, skip_x])
                skip_x = x  # Update skip_x to current x for the next potential skip connection

            if self.dropout > 0:
                raise UserWarning("Dropout not applicable to PANN")

        if self.output_dims != self.input_dims or self.layer_scaling_factor != 1:  # Need a dense layer if dimensions are different
            x = kl.Flatten(data_format=self.data_format)(x)
            out = kl.Dense(np.prod(self.output_dims), activation="linear", name="prior")(x)
            out = kl.Reshape(self.output_dims)(out)
        else:
            out = x

        return inp, out
