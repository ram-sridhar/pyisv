import numpy as np
import copy

# Copyright (C) 2023 Rama Sesha Sridhar Mantripragada. All rights reserved.
# This file is part of pyISV.


def index_agreement(s, o):
    """
    This function calculates the Index of Agreement (IOA) between two sets of data, specifically for numerical values. 
    The IOA is a statistical measure used to assess the agreement between two datasets, with a value ranging from 0 to 1. 
    An IOA of 0 indicates no agreement, while an IOA of 1 indicates perfect agreement between the datasets.

    Parameters:
    s (array-like): The first dataset (e.g., an array or list of numerical values).
    o (array-like): The second dataset (e.g., an array or list of numerical values).

    Returns:
    float: The Index of Agreement (IOA) between the two datasets, ranging from 0 to 1.

    Example usage:
    s = [1, 2, 3, 4, 5]
    o = [1.5, 2.5, 3, 4.2, 4.8]
    ia = index_agreement(s, o)
    print(ia) # Output: 0.920
    """

    ia = 1 - (np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))

    return ia


def root_mean_squared_error(y_actual, y_predicted):
    """
    Calculate the Root Mean Squared Error (RMSE) between the actual and predicted values.

    Args:
    y_actual (list or numpy array): The actual values.
    y_predicted (list or numpy array): The predicted values.

    Returns:
    float: The Root Mean Squared Error (RMSE).
    """
    # Convert inputs to numpy arrays if they are not already
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Calculate the mean squared error
    mse = np.mean((y_actual - y_predicted) ** 2)

    # Calculate the root mean squared error
    rmse = np.sqrt(mse)

    return rmse


def r2_score(y_actual, y_predicted):
    """
    Calculate the R-squared (R2) score between the actual and predicted values.

    Args:
    y_actual (list or numpy array): The actual values.
    y_predicted (list or numpy array): The predicted values.

    Returns:
    float: The R-squared (R2) score.
    """
    # Convert inputs to numpy arrays if they are not already
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)

    # Calculate the mean of actual values
    y_mean = np.mean(y_actual)

    # Calculate the total sum of squares (proportional to the variance of the data)
    total_sum_of_squares = np.sum((y_actual - y_mean) ** 2)

    # Calculate the residual sum of squares
    residual_sum_of_squares = np.sum((y_actual - y_predicted) ** 2)

    # Calculate the R-squared (R2) score
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)

    return r2


def calc_save_rmse_ioa_text(ypred, yactual):
    """
    Save the Root Mean Squared Error (RMSE) and Index of Agreement (IOA) between the predicted and actual values to text files.

    Args:
    ypred (xarray.DataArray): The predicted values.
    yactual (xarray.DataArray): The actual values.
    """
    # Create a copy of yactual with the same structure, filled with NaN values
    pred = copy.deepcopy(yactual)
    pred[:] = np.nan
    
    # Replace the NaN values in 'pred' with the predicted values from 'ypred'
    pred[:] = ypred
    
    # Unstack the multi-level index (lat, lon) for both predicted and actual values
    pred = pred.unstack()
    yactual = yactual.unstack()
    
    # Calculate the mean values across latitudes and longitudes for both predicted and actual values
    pred = pred.mean({'lat','lon'}).values
    yactual = yactual.mean({'lat','lon'}).values
    
    # Calculate the Root Mean Squared Error (RMSE) between the predicted and actual values
    rmse = root_mean_squared_error(yactual, pred)
    
    # Calculate the Index of Agreement (IOA) between the predicted and actual values
    ioa = index_agreement(pred, yactual)

    # Append the RMSE value to the "olr_rmse.txt" file
    with open("olr_rmse.txt", "a") as f:
        f.write(f"{rmse}\n")

    # Append the IOA value to the "olr_ioa.txt" file
    with open("olr_ioa.txt", "a") as f:
        f.write(f"{ioa}\n")
