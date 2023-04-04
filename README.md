# pyisv
A Machine Learning Technique to Extract the Intra Seasonal Variability (ISV) from the Geophysical Data

Dependencies:

1. proplot
2. xarray
3. tensorflow
4. netcdf4

### Installation Instructions 
#### Replace newenv with the your environemnt

conda create -n newenv python=3.10.5

conda activate newenv

pip install git+https://github.com/proplot-dev/proplot.git

conda install -c anaconda netcdf4

conda install tensorflow

conda install -c conda-forge jupyterlab

python -m ipykernel install --user --name=newenv
