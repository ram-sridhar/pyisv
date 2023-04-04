
pyisv
=============

A Machine Learning Technique to Extract the Intra Seasonal Variability (ISV) from the Geophysical Data.
=============

|doi|

Documentation
=============


Installation
============

pyisv is published on `PyPi <https://pypi.org/project/pyisv/>`__

.. code-block:: bash

   pip install pyisv

Please install the dependencies before installing the pysiv

Dependencies
============

1. proplot
2. xarray
3. tensorflow
4. netcdf4

Installation Instructions for Dependencies
============

Replace newenv with the your environemnt

.. code-block:: bash

   conda create -n newenv python=3.10.5

.. code-block:: bash

   conda activate newenv

.. code-block:: bash

   pip install git+https://github.com/proplot-dev/proplot.git

.. code-block:: bash

   conda install -c anaconda netcdf4

.. code-block:: bash

   conda install tensorflow

.. code-block:: bash

   conda install -c conda-forge jupyterlab

.. code-block:: bash

   python -m ipykernel install --user --name=newenv

.. |doi| image:: https://zenodo.org/badge/623253615.svg
   :alt: doi
   :target: https://zenodo.org/badge/latestdoi/623253615
   
