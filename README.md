<p align="center">
A Two-Asset Savings Model with an Income-Contribution Scheme
</p>

<p align="center">
Mateo Velásquez-Giraldo
</p>

<p align="center">
mvelasq2@jhu.edu
</p>

<p align="center">
Johns Hopkins University
</p>

Cite this repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4977915.svg)](https://doi.org/10.5281/zenodo.4977915)

Launch a demonstration: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Mv77/RiskyContrib/main?filepath=Code%2FPython%2FRiskyContrib.ipynb)

### Summary

This repository presents a two-asset consumption-savings model and serves as the documentation of an open-source implementation of methods to solve and
simulate it in the [HARK](https://econ-ark.org/toolkit) toolkit. The model represents an agent who can save using two different assets---one risky and
the other risk-free---to insure against fluctuations in his income, but faces frictions to transferring funds between assets. The flexibility of its
implementation and its inclusion in the HARK toolkit will allow users to adapt the model to realistic life-cycle calibrations, and also to embed it in
heterogeneous-agents macroeconomic models.

### Contents

- The main document of the repository is [`./RiskyContrib.pdf`](https://github.com/Mv77/RiskyContrib/blob/main/RiskyContrib.pdf).
- A shorter jupyter notebook with the main results can be found in `Code/Python/RiskyContrib.ipynb` and can be launched live by clicking the following badge [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Mv77/RiskyContrib/main?filepath=Code%2FPython%2FRiskyContrib.ipynb). 

### Replicating the results

To reproduce all the results in [`./RiskyContrib.pdf`](https://github.com/Mv77/RiskyContrib/blob/main/RiskyContrib.pdf) you can

##### Use [nbreproduce](https://github.com/econ-ark/nbreproduce) (requires Docker to be installed on the machine).

```
# Clone this repository
$ git clone https://github.com/Mv77/RiskyContrib

# Change working directory to RiskyContrib
$ cd RiskyContrib

# Install nbreproduce
$ pip install nbreproduce

# Reproduce all results using nbreproduce
$ nbreproduce
```

##### Install a local conda environment and execute a script that generates all the results.

```
$ conda env create -f environment.yml
$ conda activate RiskyContrib
# execute the script to create figures
$ ipython do_ALL.py
```

### Bibliographic Information

BibTex entry
```
@software{mateo_velasquez_giraldo_2021_4977915,
  author       = {Mateo Velásquez-Giraldo},
  title        = {{Mv77/RiskyContrib: A Two-Asset Savings Model with 
                   an Income-Contribution Scheme}},
  month        = jun,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.4977915},
  url          = {https://doi.org/10.5281/zenodo.4977915}
}
```
