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


class MinimalLossSaveModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath: str, grace_period: int = 0, min_eps_percent: float = 1e-3, monitor='val_loss',
                 verbose=0, save_min_val_loss='min_val_loss.txt'):
        """
        Save the model with the best loss whenever the loss is minimal
        :param filepath: path to save the model
        :param grace_period: period to wait before saving the model
        :param min_eps_percent: minimum improvement percentage since last save
        :param monitor: metric to monitor
        :param verbose: verbosity
        :param save_min_val_loss: string for filename to save minimum val loss to, else None (no saving)
        """
        super(MinimalLossSaveModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.grace_period = grace_period
        self.min_eps_percent = min_eps_percent
        self.verbose = verbose
        self.monitor = monitor
        self.best = 1e9  # initialize as a very big number
        self.save_min_val_loss = save_min_val_loss

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        current = logs.get(self.monitor)

        if current is not None:
            if epoch > self.grace_period:
                if current < self.best - self.best * self.min_eps_percent:  # Check if current loss is better than the best recorded loss by a certain percentage
                    self.best = current  # Update the best recorded loss
                    self.model.save_weights(self.filepath.format(epoch=epoch))
                    if self.verbose == 1:
                        print(
                            f'\nEpoch {epoch + 1}: saving model to {self.filepath.format(epoch=epoch)} because of improved {self.monitor}.')
                    if self.save_min_val_loss is not None:
                        with open(os.path.join(os.path.dirname(self.filepath), self.save_min_val_loss), 'a+') as f:
                            f.write(f"Minimum validation loss at epoch {epoch}: {current}\n")
                elif self.verbose == 1:
                    print(f'\nEpoch {epoch + 1}: Not saving model. {self.monitor} did not improve.')
