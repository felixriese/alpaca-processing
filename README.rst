.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3871459.svg
   :target: https://doi.org/10.5281/zenodo.3871459
   :alt: Zenodo

Processing Scripts of the ALPACA Dataset
========================================

This repository [1] includes various processing scripts of the ALPACA
dataset [2] for hyperspectral and soil moisture data. The studies are published
in [3].

:License:
    `3-Clause BSD license <LICENSE>`_

:Author:
    `Felix M. Riese <mailto:github@felixriese.de>`_

:Requirements:
    Python 3 with these `packages <requirements.txt>`_

:Citation:
    see `Citation`_ and in the `bibtex <bibliography.bib>`_ file


Workflow
----------

1. `Process Soil Moisture Data <py/1_Process_SoilMoistureData.ipynb>`_
2. `Process Full Dataset <py/2_Process_FullDataset.ipynb>`_
3. `Dataset Shift Detection <py/3_DatasetShiftDetection.ipynb>`_
4. `Estimation (Original Data) <py/4_Estimation_OriginalData.ipynb>`_
5. `Estimation (Monte Carlo) <py/5_Estimation_MonteCarloAugmentedData.ipynb>`_
6. `Plots <py/6_PlotData.ipynb>`_



----

Citation
--------

**Code:**

[1] F. M. Riese, "Processing Scripts for the ALPACA Dataset," Zenodo, 2020.
`DOI:10.5281/zenodo.3871459 <https://doi.org/10.5281/zenodo.3871459>`_

.. code:: bibtex

    @misc{riese2020processing,
        author = {Riese, Felix~M.},
        title = {{Processing Scripts for the ALPACA Dataset}},
        year = {2020},
        doi = {10.5281/zenodo.3871459},
        publisher = {Zenodo},
    }

**Dataset:**

[2] F. M. Riese, S. Schroers, J. Wienhöfer, and S. Keller "Aerial Peruvian
Andes Campaign (ALPACA) Dataset 2019," KITopen, 2020.
`DOI:10.5445/IR/1000118082 <https://doi.org/10.5445/IR/1000118082>`_

.. code:: bibtex

    @misc{riese2020aerial,
        author = {Riese, Felix~M. and Schroers, Samuel and Wienh{\"o}fer, Jan and Keller, Sina},
        title = {Aerial Peruvian Andes Campaign (ALPACA) Dataset 2019},
        year = {2020},
        doi = {10.5445/IR/1000118082},
        organization = {KITopen},
    }


Code is Supplementary Material to
----------------------------------

[3] Felix M. Riese. "Development and Applications of Machine Learning Methods
for Hyperspectral Data." PhD thesis. Karlsruhe, Germany: Karlsruhe Institute of
Technology (KIT), 2020.

.. code:: bibtex

    @phdthesis{riese2020development,
        author = {Riese, Felix~M.},
        title = {{Development and Applications of Machine Learning Methods for Hyperspectral Data}},
        school = {Karlsruhe Institute of Technology (KIT)},
        year = {2020},
        address = {Karlsruhe, Germany},
    }
