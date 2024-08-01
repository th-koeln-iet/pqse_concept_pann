import os

import numpy as np
import tensorflow as tf


class CyclicLR(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=0.001, max_lr=0.006, step_size=2000., gamma=0.99994):
        """
        Cyclical Learning Rate (CLR) Callback
        :param min_lr: Minimum learning rate
        :param max_lr: Maximum learning rate
        :param step_size:
        :param gamma: Value for exp_range mode that determines the decay
        """
        super(CyclicLR, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        base_lr = self.min_lr + (self.max_lr - self.min_lr) * np.maximum(0, (1 - x))
        return base_lr

    def on_train_begin(self, logs=None):
        logs = logs or {}
        initial_lr = self.clr()
        tf.keras.backend.set_value(self.model.optimizer.lr, initial_lr)
        logs['lr'] = initial_lr  # Log the initial learning rate

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        new_lr = self.clr()
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        logs['lr'] = new_lr  # Update the learning rate in logs at the end of each batch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Ensure the latest learning rate is logged at the end of each epoch
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        logs['lr'] = current_lr
