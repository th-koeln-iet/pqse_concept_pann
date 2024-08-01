import tensorflow as tf
import numpy as np
from models.DNN import DNN


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, input_len, key_dim):
        super(PositionalEncodingLayer, self).__init__()
        self.positional_encoding = self.get_positional_encoding(input_len, key_dim)

    def get_positional_encoding(self, input_len, key_dim):
        pos = np.arange(input_len)[:, np.newaxis]
        i = np.arange(key_dim)[np.newaxis, :]
        angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(key_dim))

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding


class SharedEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, dropout_rate=0.1):
        super(SharedEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation='relu'),
                                        tf.keras.layers.Dense(key_dim)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class Transformer(DNN):
    def __init__(self, hyperparams, network_data, callbacks=None):
        super(Transformer, self).__init__(hyperparams, network_data, callbacks)
        self.num_heads = hyperparams['num_heads']
        self.key_dim = hyperparams['key_dim']
        self.ff_dim = hyperparams['ff_dim']
        self.num_layers = hyperparams['num_hidden_layers']
        self.dropout_rate = hyperparams['dropout']

    def create_layers(self):
        inp = tf.keras.layers.Input(shape=self.input_dims)
        x = tf.keras.layers.Flatten()(inp)
        x = tf.keras.layers.Dense((self.output_dims[0] * self.input_dims[1] * self.input_dims[2]))(x)
        x = tf.keras.layers.Reshape((self.output_dims[0], self.input_dims[1] * self.input_dims[2]))(x)
        x = PositionalEncodingLayer(self.output_dims[0], self.key_dim)(x)
        for _ in range(self.num_layers):
            x = SharedEncoderLayer(self.num_heads, self.key_dim, self.ff_dim, self.dropout_rate)(x)
        x = tf.keras.layers.Flatten()(x)
        output_neurons = np.prod(self.output_dims)
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
