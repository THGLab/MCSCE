MCSCE - Sidechain packing library
=================================

.. image:: https://img.shields.io/badge/version-v0.0.0-orange
    :target:https://github.com/THGLab/MCSCE
    :alt: Version

.. image:: https://github.com/THGLab/MCSCE/workflows/Tests/badge.svg?branch=master
    :target: https://github.com/THGLab/MCSCE/actions?workflow=Tests
    :alt: Test Status

.. image:: https://github.com/THGLab/MCSCE/workflows/Package%20Build/badge.svg?branch=master
    :target: https://github.com/THGLab/MCSCE/actions?workflow=Package%20Build
    :alt: Package Build

.. image:: https://codecov.io/gh/THGLab/MCSCE/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/THGLab/MCSCE
    :alt: Codecov

.. image:: https://img.shields.io/codacy/grade/ae042ac066554bfab398024b0beea6a5/master?label=Codacy
    :target: https://app.codacy.com/manual/THGLab/MCSCE/dashboard
    :alt: Codacy

.. image:: https://api.codeclimate.com/v1/badges/d96cc9a1841a819cd4f5/maintainability
   :target: https://codeclimate.com/github/THGLab/MCSCE/maintainability
   :alt: Maintainability

.. image:: https://img.shields.io/codeclimate/tech-debt/THGLab/MCSCE
    :target: https://codeclimate.com/github/THGLab/MCSCE
    :alt: Code Climate technical debt

.. image:: https://img.shields.io/readthedocs/MCSCE/latest?label=Read%20the%20Docs
    :target: https://MCSCE.readthedocs.io/en/latest/index.html
    :alt: Read the Docs

Monte Carlo Side Chain Entropy package for generating side chain packing for
fixed protein backbone.

References
==========

1.Lin, M. S., Fawzi, N. L. & Head-Gordon, T. Hydrophobic Potential of Mean Force
as a Solvation Function for Protein Structure Prediction. Structure 15, 727–740
(2007).

2. Bhowmick, A. & Head-Gordon, T. A Monte Carlo Method for Generating Side Chain
Structural Ensembles. Structure 23, 44–55 (2015).

How to Install
==============

1. Clone this repository::

    git clone https://github.com/THGLab/MCSCE

2. Navigate to the new folder::

    cd MCSCSE

3. Create a dedicated Conda environment with the needed dependencies::

    conda env create -f requirements.yml

4. Install MCSCE package manually::

    python setup.py develop --no-deps

5. To update to the latest version::

    git pull

How to Use
==========

In your terminal window run for help::

    mcsce -h

How to Contribute
=================

Contribute to this project following the instructions in
`docs/CONTRIBUTING.rst`_ file.

.. _docs/CONTRIBUTING.rst: https://github.com/THGLab/MCSCE/blob/master/docs/CONTRIBUTING.rst
