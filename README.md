

# **IEVI**: Information-Efficient Variational Inference for SDEs

------------------------------------------------------------------------

## What is **IEVI**?

**IEVI** is a variational inference method for SDEs that approximates
the intractable latent posterior. Compared to other variational
inference methods, **IEVI** captures not only the correct conditional
dependence structure but also the correct *information flow* from the
observations to the intractable posterior.

------------------------------------------------------------------------

## Installation

This will clone the repo into a subfolder `sde-ievi`, from where you (i)
issue the `git clone` command and (ii) install the package from source.

``` bash
git clone https://github.com/mlysy/sde-ievi
cd sde-ievi
pip install .
```

## Building Documentation

From within `sde-ievi/docs`:

``` bash
quartodoc build
quarto render
```
