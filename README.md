# infer_structcol

infer_structcol
--------

This is a package created for the PHY 201 class, Spring of 2017 at Harvard University. It uses bayesian inference to estimate parameters of structural color samples made out of colloidal packings. It imports the structcol package to do scattering calculations. It includes a generative model to account for experimental uncertainties such as the effect of sample chambers on the reflectance spectra of structurally-colored samples.

This package uses the structural-color package to calculate reflectance spectra using a single scattering model. Before you install the infer_structcol package, you will need to git clone or download the structural-color package from https://github.com/manoharan-lab/structural-color and run 'python setup.py install' to install the package. If the installation does not work, you can manually add the directory of the package to your python path.

Input data files should contain spectra consisting of wavelength, reflectance/transmittance, and uncertainty at each wavelength. The package also includes a convenient conversion tool that calculates reflectance/transmittance values and uncertainties from an ensemble of identical measurements, along with appropriate control runs.
