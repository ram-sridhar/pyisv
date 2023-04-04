conda create -n newenv python=3.10.5

conda activate newenv

pip install git+https://github.com/proplot-dev/proplot.git

conda install -c anaconda netcdf4

conda install tensorflow

conda install -c conda-forge jupyterlab

python -m ipykernel install --user --name=package

