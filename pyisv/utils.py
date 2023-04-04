import importlib
import numpy as np
import xarray as xr
import copy

# This file is part of pyISV.

def check_if_packages_installed():
    """
    This tests whether required dependices are installed or not and prints the dependicies 
    that are not installed.
    
    """
    
    # Define the required packages
    required_packages = ['numpy','xarray','tensorflow','proplot','matplotlib','scipy','pandas','logging','multiprocessing']

    # Loop through the required packages and check if they are installed
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)

    # If there are missing packages, recommend the user to install them
    if len(missing_packages) > 0:
        print("The following packages are required but not installed: ")
        print(', '.join(missing_packages))
        print("Please install these packages before running the code.")
    else:
        print("All required packages are installed.")

        
def smooth_clim(data):   
    """    
    Smooths the input climate data by calculating daily climatology and applying a low-pass
    Fourier filter to retain only the first few harmonic components of the annual cycle.
    
    Args:
    data (xarray.DataArray or xarray.Dataset): Input climate data with a "time" dimension.
    
    Returns:
    smclim (xarray.DataArray or xarray.Dataset): Smoothed climatology data with the same 
    dimensions and coordinates as the input data.
    
    Notes:
    This function assumes that the input data is daily data and is evenly spaced in time.
    The number of harmonics to retain is determined by the variable `nharm`. In this
    implementation, `nharm` is set to 3.
    """
    
    # Calculate daily climatology
    clim = data.groupby("time.dayofyear").mean("time")
    # smoothed annual cycle
    nharm = 3
    fft = np.fft.rfft(clim, axis=0)
    fft[nharm] = 0.5*fft[nharm]
    fft[nharm+1:] = 0
    dataout = np.fft.irfft(fft, axis=0)
    smclim = copy.deepcopy(clim)
    smclim[:] = dataout 
    
    return smclim


def filtwghts_lanczos(nwt, filt_type, fca, fcb):
    """    
    Calculates the Lanczos filter weights.
    
    Parameters
    ----------
    nwt : int
        The number of weights.
    filt_type : str
        The type of filter. Must be one of 'low', 'high', or 'band'.
    fca : float
        The cutoff frequency for the low or band filter.
    fcb : float
        The cutoff frequency for the high or band filter.
    
    Returns
    -------
    w : ndarray
        The Lanczos filter weights.
    
    Notes
    -----
    The Lanczos filter is a type of sinc filter that is truncated at a specified frequency.
    This function implements a Lanczos filter in the time domain.
    """
    
    pi = np.pi
    k = np.arange(-nwt, nwt+1)
    
    if filt_type == 'low':
        w = np.zeros(nwt*2+1)
        w[:nwt] = ((np.sin(2 * pi * fca * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt))
        w[nwt+1:] = ((np.sin(2 * pi * fca * k[nwt+1:]) / (pi * k[nwt+1:])) * np.sin(pi * k[nwt+1:] / nwt) / (pi * k[nwt+1:] / nwt))
        w[nwt] = 2 * fca
    elif filt_type == 'high':
        w = np.zeros(nwt*2+1)
        w[:nwt] = -1 * (np.sin(2 * pi * fcb * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt)
        w[nwt+1:] = -1 * (np.sin(2 * pi * fcb * k[nwt+1:]) / (pi * k[nwt+1:])) * np.sin(pi * k[nwt+1:] / nwt) / (pi * k[nwt+1:] / nwt)
        w[nwt] = 1 - 2 * fcb
    else:
        w1 = np.zeros(nwt*2+1)
        w1[:nwt] = (np.sin(2 * pi * fca * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt)
        w1[nwt+1:] = (np.sin(2 * pi * fca * k[nwt+1:]) / (pi * k[nwt+1:])) * np.sin(pi * k[nwt+1:] / nwt) / (pi * k[nwt+1:] / nwt)
        w1[nwt] = 2 * fca
        w2 = np.zeros(nwt*2+1)
        w2[:nwt] = (np.sin(2 * pi * fcb * k[:nwt]) / (pi * k[:nwt])) * np.sin(pi * k[:nwt] / nwt) / (pi * k[:nwt] / nwt)
        w2[nwt+1:] = (np.sin(2 * pi * fcb * k[nwt+1:]) / (pi * k[nwt+1:])) * np.sin(pi * k[nwt+1:] / nwt) / (pi * k[nwt+1:] / nwt)
        w2[nwt] = 2 * fcb
        w = w2 - w1

    return w


def calc_anomalies(data, startYear, endYear, smooth_climatology=False):
    """
    Calculate anomalies for a given data array based on the climatology of a specified period.

    Args:
    data (xarray.DataArray): The input data array containing the variable of interest.
    startYear (str or int): The starting year for the climatology period.
    endYear (str or int): The ending year for the climatology period.
    smooth_climatology (bool, optional): Whether to smooth the climatology using a custom function (default is False).

    Returns:
    xarray.DataArray: The anomalies of the input data array based on the calculated climatology.
    """
    # Calculate the climatology for the period
    if smooth_climatology == True:
        data_clim = smooth_clim(data.sel(time=slice(startYear, endYear)))
    else:
        data_clim = data.sel(time=slice(startYear, endYear)).groupby("time.dayofyear").mean("time")

    # Calculate the anomalies by subtracting the climatology from the original data
    data_anom = data.groupby("time.dayofyear") - data_clim
    
    return data_anom, data_clim

    
    
def calc_lanczos_filtered_anomalies(data, nwgths, filt_type, tpa, tpb):
    """
    Calculate Lanczos band-pass filtered anomalies for a given dataset.
    
    Args:
        data (xarray.DataArray): Input data array.
        nwghts (int): Number of weights for the Lanczos filter.
        filt_type (str): Type of filter to be used (e.g., 'band').
        tpa (float): Upper bound of the filter period.
        tpb (float): Lower bound of the filter period.

    Returns:
        tuple: A tuple containing the unfiltered anomalies (olri_anom) and
               band-pass filtered anomalies (olri_bpf).
    """
    
    # Calculate the 30-90-day band-pass Lanczos filtered OLR anomalies
    wgths = filtwghts_lanczos(nwgths, 'band', 1/tpb, 1/tpa)
    wgths = xr.DataArray(wgths, dims=['window'])

    # Apply the Lanczos filter to the anomalies
    data_bpf = data.rolling(time=len(wgths), center=True).construct('window').dot(wgths)
    data_bpf = data_bpf.dropna(dim='time')

    return data_bpf


def calc_olr_proxy_anomalies(olrni_anom):
    """
    Calculate Outgoing Longwave Radiation (OLR) proxy anomalies.
    
    This function takes an xarray.DataArray of OLR data and computes anomalies
    by subtracting the mean of the previous 40 timesteps at each point in the array. 
    A 5-day tapered running mean is then applied to smooth the results.
    
    Parameters:
    olrni_anom (xarray.DataArray): An xarray.DataArray containing OLR data with dimensions (time, lat, lon).
    
    Returns:
    xarray.DataArray: An xarray.DataArray of the same dimensions containing the OLR proxy anomalies.
    """

    # Create a deep copy of the input data starting from the 40th timestep
    exa = copy.deepcopy(olrni_anom[40:,:,:])
    exa[:] = np.nan

    # Compute the anomalies by subtracting the mean of the previous 40 timesteps
    for i in range(exa.shape[0]):
        exa[i,:,:] = olrni_anom[40+i,:,:] - olrni_anom[i:40+i,:,:].mean('time')

    # Apply a 5-day tapered running mean to smooth the results
    exb = exa.rolling(time=5,center=True).mean()
    
    # Handle edge cases for the tapered running mean
    exb[0,:,:] = exa[0,:,:]
    exb[-1,:,:] = exa[-1,:,:]
    exb[1,:,:] = exa[0:3,:,:].mean('time')
    exb[-2,:,:] = exa[-3:,:,:].mean('time')

    return exb


def check_nan(data):
    """
    Check if the input xarray DataArray contains any NaN values.
    
    Parameters:
    data (xarray.DataArray): Input data array to check for NaN values.
    
    Returns:
    nan_indices (numpy.ndarray): Indices of NaN values in the input data array, if any.
    None: If the input data array does not contain any NaN values.
    """
    
    # Check if the data array contains any NaN values
    if data.isnull().any().item():
        # Print a message indicating the presence of NaN values
        print("DataArray contains NaN values. The input data to the CNN should not contain NaN values ")
        
        # Find the locations of NaN values
        nan_locations = data.isnull()
        
        # Get the indices of NaN values
        nan_indices = np.argwhere(nan_locations.values)
        
        # Return the indices of NaN values
        return nan_indices
    else:
        # Print a message indicating the absence of NaN values
        print("DataArray does not contain NaN values.")
        # Return None, as there are no NaN values
        return None


def read_dataset_with_warning_suppressed(eof_file):
    """
    Read a dataset from a file while suppressing SerializationWarning.

    Parameters:
    eof_file (str): The path to the dataset file.

    Returns:
    xarray.Dataset: The dataset read from the file.
    """
    
    import warnings
    from xarray.coding.times import SerializationWarning
    # Suppress SerializationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SerializationWarning)

        # Read the dataset from the file
        ds = xr.open_dataset(eof_file)

    return ds


def remove_partial_year(da):

    """
    This function removes partial years from the beginning and end of a given xarray DataArray.
    
    Parameters:
    -----------
    da : xarray.DataArray
        The input data array that contains time-series data.
        
    Returns:
    --------
    xarray.DataArray
        The updated data array with partial years removed from the beginning and end.
    """

    start_year = da.time[0].dt.year.values
    end_year = da.time[-1].dt.year.values

    da = da.sel(time=slice(f"{start_year+1}-01-01", None))
    da = da.sel(time=slice(None, f"{end_year-1}-12-31"))
        
    return da