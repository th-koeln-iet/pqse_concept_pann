import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None) -> any:
    """
    Calculate the Mean Absolute Error (MAE).
    If axis is None, calculate the MAE over all axes returning a float value.
    If axis is not None, calculate the MAE over the specified axes returning a numpy array.
    :param np.ndarray y_true: True values
    :param np.ndarray y_pred: predicted values
    :param Tuple int axis: integer tuple of axes along which to calculate the error
    :return: mean absolute error, optionally along specified axes
    """
    validate_input_shapes(y_true, y_pred, axis)

    # Calculate the Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred), axis=axis)

    return mae


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None) -> any:
    """
    Calculate the Mean Squared Error (MSE).
    If axis is None, calculate the MSE over all axes returning a float value.
    If axis is not None, calculate the MSE over the specified axes returning a numpy array.
    :param np.ndarray y_true: True values
    :param np.ndarray y_pred: predicted values
    :param Tuple int axis: integer tuple of axis along which to calculate the error
    :return: mean squared error, optionally along specified axes
    """
    validate_input_shapes(y_true, y_pred, axis)

    # Calculate the Mean Squared Error
    mse = np.mean(np.square(y_true - y_pred), axis=axis)

    return mse


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None) -> any:
    """
    Calculate the Root Mean Squared Error (RMSE).
    If axis is None, calculate the RMSE over all axes returning a float value.
    If axis is not None, calculate the RMSE over the specified axes returning a numpy array.
    :param np.ndarray y_true: True values
    :param np.ndarray y_pred: predicted values
    :param Tuple int axis: integer tuple of axis along which to calculate the error
    :return: root mean squared error, optionally along specified axes
    """
    validate_input_shapes(y_true, y_pred, axis)

    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred), axis=axis))

    return rmse


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, axis=None,
                                   mode: str = "zero_constant",
                                   eps: float = 1e-9) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE).
    If axis is None, calculate the MAPE over all axes returning a float value.
    If axis is not None, calculate the MAPE over the specified axes returning a numpy array.
    :param y_true: True values
    :param y_pred: Predicted values
    :param axis: axis along which to calculate the error
    :param mode: Mode, either "zero_constant" or "zero_masked" where zero_constant adds a small constant to the denominator
    and zero_masked masks the zero values
    :param eps: small constant to avoid division by zero
    :return: mean absolute percentage error, optionally along specified axes
    """
    validate_input_shapes(y_true, y_pred, axis)

    if mode == "zero_masked":
        # Avoid division by zero
        non_zero_mask = y_true != 0
        y_test = y_true[non_zero_mask]
        y_pred = y_pred[non_zero_mask]

        # Calculate MAPE
        ape = np.abs((y_test - y_pred) / y_test) * 100
    elif mode == "zero_constant":
        ape = np.abs((y_true - y_pred) / (y_true + eps)) * 100
    else:
        raise ValueError("mode must be either 'zero_constant' or 'zero_masked'")
    mape = np.mean(ape, axis=axis)
    return mape


def validate_input_shapes(y_true, y_pred, axis):
    # Check if both inputs are numpy arrays
    if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
        raise TypeError("Both y_true and y_pred must be NumPy arrays.")

    # Check if the shapes of the arrays are the same
    if y_true.shape != y_pred.shape:
        raise ValueError("The shapes of y_true and y_pred must be the same.")

    if axis is None:
        return

    # If axis is a single integer, convert it into a tuple
    if isinstance(axis, int):
        axis = (axis,)

    # Check if axis is a tuple or list of integers
    if not isinstance(axis, (tuple, list)) or not all(isinstance(a, int) for a in axis):
        raise ValueError("Axis must be an integer, tuple of integers, or list of integers.")

    # Check if each axis value is valid
    if not all(0 <= a < y_true.ndim for a in axis):
        raise ValueError("All axis values must be valid for the input array.")

