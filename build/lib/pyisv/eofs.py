import xarray as xr
import numpy as np
from pyisv.utils import read_dataset_with_warning_suppressed
import copy

def process_evec(ds, evec_name):
    """
    Copyright (C) 2023 Rama Sesha Sridhar Mantripragada. All rights reserved.
    This file is part of pyISV.
    
    Process eigenvectors.
    """
    evec = ds[evec_name].sel(lat=slice(-30, 30))
    evec[0, :, :] = -evec[0, :, :]
    
    return evec.stack(grid=("lat", "lon")).transpose("time", "grid")


def pcs_lag_kikuchi(eof_file, data):
    """
    Calculate the principal components (PCs) of Kikuchi EOFs for given data.

    Parameters:
    eof_file (str): Path to the EOF file containing eigenvectors.
    
    data (xarray.DataArray): Input data array to calculate PCs for.

    Returns:
    pcs1, pcs2, pcs3 (tuple): Calculated lagged principal components.
    """
    ds = read_dataset_with_warning_suppressed(eof_file)

    # Process eigenvectors
    evec0_rg, evec1_rg, evec2_rg = (process_evec(ds, name) for name in ("evec0", "evec1", "evec2"))
    
    # Reshape data
    data = data.stack(grid=("lat","lon")).transpose("time","grid")
    
    # Shift data by 10, 5, and 0 days
    ex1, ex2, ex3 = (data.shift(time=t) for t in (10, 5, 0))

    # Calculate PCs
    pcs1, pcs2, pcs3 = (ex.data @ evec_rg.T.data for ex, evec_rg in zip((ex1, ex2, ex3), (evec0_rg, evec1_rg, evec2_rg)))

    # Remove the first 10 rows
    pcs1, pcs2, pcs3 = (pcs[10:, :] for pcs in (pcs1, pcs2, pcs3))

    # Convert to xarray DataArray and assign time coordinate
    pcs1, pcs2, pcs3 = (xr.DataArray(pcs, dims=['time', 'pc'], coords={"time": data.time[10:]}) for pcs in (pcs1, pcs2, pcs3))

    return pcs1, pcs2, pcs3


def pcs_kikuchi(eof_file, data):
    """
    Calculate standardized principal components (PCs) following Kikuchi et al. 2012
    
    Parameters:
    eof_file (str): Path to the EOF file containing eigenvectors.
    
    data (xarray.DataArray): Input OLR to calculate PCs for.
    
    Returns:
    pcs (xarray.DataArray): standardized principal components.
    """
    # Obtain individual principal components with lags
    pcs1, pcs2, pcs3 = pcs_lag_kikuchi(eof_file, data)
    
    # Sum the three PCs
    pcs = pcs1 + pcs2 + pcs3

    # Standardize the PCs
    pcs_std = np.expand_dims(np.nanstd(pcs, axis=0), axis=0)
    pcs /= pcs_std
    
    return pcs


def reconstruct_mjo_kikuchi(eof_file, data):
    """
    Reconstruct the MJO OLR using the EOF file and OLR following Kikuchi et al. 2012
    
    Parameters:
    eof_file (str): Path to the EOF file containing eigenvectors.
    
    data (xarray.DataArray): Input OLR.
    
    Returns:
    data (xarray.DataArray): Reconstructed MJO OLR.
    """
    # Open the dataset containing eigenvectors
    ds = read_dataset_with_warning_suppressed(eof_file)
    
    # Process eigenvectors
    evec0_rg, evec1_rg, evec2_rg = (process_evec(ds, name) for name in ("evec0", "evec1", "evec2"))
    
    # Obtain principal components with lags
    pcs1, pcs2, pcs3 = pcs_lag_kikuchi(eof_file, data)
    
    # Calculate the reconstructed data for each eigenvector and principal component
    tt1 = np.expand_dims(evec0_rg.T[:,0],axis=1) @ np.expand_dims(pcs1.T[0,:],axis=0) + np.expand_dims(evec0_rg.T[:,1],axis=1) @ np.expand_dims(pcs1.T[1,:],axis=0)
    tt2 = np.expand_dims(evec1_rg.T[:,0],axis=1) @ np.expand_dims(pcs2.T[0,:],axis=0) + np.expand_dims(evec1_rg.T[:,1],axis=1) @ np.expand_dims(pcs2.T[1,:],axis=0)
    tt3 = np.expand_dims(evec2_rg.T[:,0],axis=1) @ np.expand_dims(pcs3.T[0,:],axis=0) + np.expand_dims(evec2_rg.T[:,1],axis=1) @ np.expand_dims(pcs3.T[1,:],axis=0)

    # Sum the reconstructed data
    tt = tt1 + tt2 + tt3

    # Reshape data
    data = data.stack(grid=("lat","lon")).transpose("time","grid")
    
    # Create a deep copy of the input data without the first 10 rows
    dat = copy.deepcopy(data[10:,:])
    
    # Replace the input data with the reconstructed data
    datT = dat.T
    datT[:] = np.nan
    datT[:] = tt
    data = datT.unstack()
    
    return data