
pyisv
=============

A Machine Learning Technique to Extract the Intra Seasonal Variability (ISV) from the Geophysical Data.
=============

|doi|

Documentation
=============


Installation
============

Proplot is published on `PyPi <https://pypi.org/project/pyisv/>`__

.. code-block:: bash

   pip install proplot

Dependencies
============

1. proplot
2. xarray
3. tensorflow
4. netcdf4

Installation Instructions for Dependencies
============

Replace newenv with the your environemnt

conda create -n newenv python=3.10.5

conda activate newenv

pip install git+https://github.com/proplot-dev/proplot.git

conda install -c anaconda netcdf4

conda install tensorflow

conda install -c conda-forge jupyterlab

python -m ipykernel install --user --name=newenv

.. |doi| image:: https://zenodo.org/badge/623253615.svg
   :alt: doi
   :target: https://zenodo.org/badge/latestdoi/623253615
   
