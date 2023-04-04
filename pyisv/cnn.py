import os
import logging
import copy
import psutil
import multiprocessing
import numpy as np
import tensorflow as tf
    
#Copyright (C) 2023 Rama Sesha Sridhar Mantripragada. All rights reserved.
#This file is part of pyISV.
    
def check_system_capabilities():
    
    """
    This function checks the system's hardware capabilities, specifically the presence of GPUs.
    If GPUs are detected, it enables memory growth for TensorFlow to allocate only as much GPU memory as needed.
    If no GPUs are detected, it prints the system's CPU and memory information.
    Raises:
        RuntimeError: If memory growth cannot be set after GPUs have been initialized.
    """

    # Check for the presence of GPU devices
    if tf.config.list_physical_devices('GPU'):
        try:
            # Print that the system has GPU capabilities
            print("The system has GPU capabilities.")

            # List all the available GPUs
            gpus = tf.config.list_physical_devices('GPU')

            # Enable memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # List logical GPU devices
            logical_gpus = tf.config.list_logical_devices('GPU')

            # Print the number of detected GPUs
            print("Detected GPUs:", len(gpus))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        # If no GPUs are detected, print CPU and memory information
        print("The system has only CPU capabilities.")

        # Print the number of CPU cores
        print("Number of cores:", multiprocessing.cpu_count())

        # Get the system's virtual memory information
        memory = psutil.virtual_memory()

        # Print the total, available, used memory, and percentage of memory used
        print("Total memory:", memory.total / (1024 ** 3), "GB")
        print("Available memory:", memory.available / (1024 ** 3), "GB")
        print("Used memory:", memory.used / (1024 ** 3), "GB")
        print("Percentage of memory used:", memory.percent, "%")
        

def split_dataset_cnn(startTrain, endTrain, startVal, endVal, startTest, endTest, olri_anom, olri_bpf, num_cpus):
    """
    Split the dataset into training, validation, and testing sets for a Convolutional Neural Network (CNN).
    
    Args:
        startTrain (str): Start date for the training set.
        endTrain (str): End date for the training set.
        startVal (str): Start date for the validation set.
        endVal (str): End date for the validation set.
        startTest (str): Start date for the testing set.
        endTest (str): End date for the testing set.
        olri_anom (xarray.DataArray): Anomaly data array.
        olri_bpf (xarray.DataArray): Band-pass filtered data array.
        num_cpus (int): Number of CPU cores to use for parallel processing.

    Returns:
        tuple: A tuple containing the training, validation, and testing sets, along with the number of grid points.
    """
    # Combine latitude and longitude coordinates into a single "grid" dimension
    # and transpose the data to (time, grid)
    olri_anom_rg = olri_anom.stack(grid=("lat", "lon")).transpose("time", "grid")
    olri_bpf_rg = olri_bpf.stack(grid=("lat", "lon")).transpose("time", "grid")

    # Split the dataset into training, validation, and testing sets
    xtrain = olri_anom_rg.sel(time=slice(startTrain, endTrain)).expand_dims(dim='new', axis=0).values
    ytrain = olri_bpf_rg.sel(time=slice(startTrain, endTrain)).expand_dims(dim='new', axis=0).values

    xval = olri_anom_rg.sel(time=slice(startVal, endVal)).expand_dims(dim='new', axis=0).values
    yval = olri_bpf_rg.sel(time=slice(startVal, endVal)).expand_dims(dim='new', axis=0).values

    xtest = olri_anom_rg.sel(time=slice(startTest, endTest)).expand_dims(dim='new', axis=0).values
    ytest = olri_bpf_rg.sel(time=slice(startTest, endTest))

    # Split the data along the grid dimension for parallel processing across CPU cores
    xtrain_split = np.array_split(xtrain, num_cpus, axis=2)
    ytrain_split = np.array_split(ytrain, num_cpus, axis=2)
    xval_split = np.array_split(xval, num_cpus, axis=2)
    yval_split = np.array_split(yval, num_cpus, axis=2)
    xtest_split = np.array_split(xtest, num_cpus, axis=2)

    # Calculate the number of grid points for each CPU core
    ngp = [xtrain_split[i].shape[2] for i in range(num_cpus)]

    return xtrain_split, ytrain_split, xval_split, yval_split, xtest_split, ytest, ngp


def cnn_bpf(cc, xtrain_split, ytrain_split, xval_split, yval_split, xtest_split, no_epochs, verbosity, kernel1, kernel2, ngp, outPath):
    
    """
    Trains a convolutional neural network model on the input training data and returns a bandpass filtered test data.
    
    Args:
        cc (int): Index of the parallel processing.
        
        xtrain_split (numpy array): Input training data for the current processor.
        
        ytrain_split (numpy array): Target training data for the current processor.
        
        xval_split (numpy array): Input validation data for the current processor.
        
        yval_split (numpy array): Target validation data for the current processor.
        
        xtest_split (numpy array): Input test data for the current processor.
        
        no_epochs (int): Number of epochs to train the model.
        
        verbosity (int): Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
        
        kernel1 (int): Size of the kernel for the first depthwise convolution layer.
        
        kernel2 (int): Size of the kernel for the second depthwise convolution layer.
        
        ngp (int): Number of grid points in the input data.
        
        outPath (str): Output path to save the model weights.
        
    Returns:
        numpy array: Predictions on the test data for the current processor.
    """
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
    inputs = tf.keras.Input(shape=(None,ngp[cc]),batch_size=1,name='input_layer')
    smoth1 = tf.keras.layers.DepthwiseConv1D(kernel_size=kernel1,padding='same',use_bias=False,activation='linear')(inputs)
    diff = tf.keras.layers.subtract([inputs, smoth1])
    smoth2 = tf.keras.layers.DepthwiseConv1D(kernel_size=kernel2,padding='same',use_bias=False,activation='linear')(diff)
    model = tf.keras.Model(inputs=inputs, outputs=smoth2)
    model.compile(optimizer='adam', loss='mse')
    model.fit(xtrain_split[cc],ytrain_split[cc],epochs=no_epochs,validation_data=(xval_split[cc], yval_split[cc]),verbose=verbosity,callbacks=[callback])
    
    if outPath != None:
        model.save_weights(outPath+str(cc)+'.h5')
        
    pred = model.predict(xtest_split[cc]).squeeze()
    
    return pred


def wrapper_cnn_bpf(args):
    return cnn_bpf(*args)


def save_cnn_weights_text(kernel1, kernel2, ngp, var_name,outPath):
    
    """
    Copyright (C) 2023 Rama Sesha Sridhar Mantripragada. All rights reserved.
    This file is part of pyISV.
    
    Loads saved CNN model weights and saves the filter weights to text files.
    
    Args:
        kernel1 (int): Size of the kernel for the first depthwise convolution layer.
        
        kernel2 (int): Size of the kernel for the second depthwise convolution layer.
        
        ngp (list): List of the number of grid points for each input data.
        
        var_name (str): Variable name to use for the output text files.
        
    Returns:
        None
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    filter1 = []
    filter2 = []

    for i, ng in enumerate(ngp):
        inputs = tf.keras.Input(shape=(None,ng),batch_size=1,name='input_layer')
        smoth1 = tf.keras.layers.DepthwiseConv1D(kernel_size=kernel1,padding='same',use_bias=False,activation='linear')(inputs)
        diff = tf.keras.layers.subtract([inputs, smoth1])
        smoth2 = tf.keras.layers.DepthwiseConv1D(kernel_size=kernel2,padding='same',use_bias=False,activation='linear')(diff)
        model = tf.keras.Model(inputs=inputs, outputs=smoth2)
        model.compile(optimizer='adam', loss='mse')
        model.load_weights(outPath + str(i) + '.h5')
        ex1 = model.layers[1].get_weights()[0].squeeze()
        ex2 = model.layers[3].get_weights()[0].squeeze()
        filter1.append(ex1)
        filter2.append(ex2)
        
    filter1 = np.hstack(filter1)
    filter2 = np.hstack(filter2)
    
    np.savetxt(outPath+var_name+'.filter1.txt', filter1, delimiter=',')
    np.savetxt(outPath+var_name+'.filter2.txt', filter2, delimiter=',')


def calc_anomalies_from_cnn_weights(filter1, filter2, olrni_anom):
    """
    Calculate CNN (Convolutional Neural Network) anomalies of OLR data.

    :param filter1: Path to a .txt file containing the first filter coefficients (low-pass filter).
    :type filter1: str
    :param filter2: Path to a .txt file containing the second filter coefficients (high-pass filter).
    :type filter2: str
    :param olrni_anom: An xarray.DataArray containing OLR data with dimensions (time, lat, lon).
    :type olrni_anom: xarray.DataArray

    :return: An xarray.DataArray of the same dimensions containing the CNN anomalies.
    :rtype: xarray.DataArray

    This function applies two filters provided as input (filter1 and filter2) to the
    Outgoing Longwave Radiation (OLR) data. First, a low-pass filter is applied using
    filter1, then a high-pass filter is applied by subtracting the low-pass filtered
    data from the original data. Finally, filter2 is applied to the high-pass filtered data.
    """

    # Load filter coefficients from input files
    filt1 = np.loadtxt(filter1, delimiter=',')
    filt2 = np.loadtxt(filter2, delimiter=',')

    # Stack and transpose the input OLR data
    olrni_anom_rg = olrni_anom.stack(grid=("lat", "lon")).transpose("time", "grid")

    # Create a deep copy of the input data for the output
    olrni_cnn_rg = copy.deepcopy(olrni_anom_rg)
    olrni_cnn_rg[:] = np.nan

    # Apply the filters to the OLR data
    for i in range(filt1.shape[1]):
        LowPass = np.convolve(olrni_anom_rg[:, i], filt1[:, i], mode='same')
        HighPass = olrni_anom_rg[:, i] - LowPass
        olrni_cnn_rg[:, i] = np.convolve(HighPass, filt2[:, i], mode='same')

    # Unstack the filtered data
    olrnicnn = olrni_cnn_rg.unstack()

    return olrnicnn