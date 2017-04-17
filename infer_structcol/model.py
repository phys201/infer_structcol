import numpy as np
from .main import rescale

def calc_prior(theta, phi_guess):
    vol_frac, l0, l1 = theta

    if l0 < 0 or l0 > 1 or l0+l1 <0 or l0+l1 > 1:
        # Losses are not in range [0,1] for some wavelength
        return -np.inf 

    if vol_frac < .35 or vol_frac < .73:
        # Outside range of validity of multiple scattering model
        return -np.inf
    
    var = .0025 # based on expected normal range
    return np.exp(-(vol_frac-phi_guess)**2/var)

def calc_likelihood(spect_theory, spect_meas, sigma_theory, sigma_meas, wavelength, l0, l1):
    """
    returns likelihood of obtaining an experimental dataset from a given spectrum
    
    Parameters:
        spect_theory: a spectrum calculated from the MC model (array of length N)
        spect_meas: dependent variable - experimental measurements (array of length N)
        sigma_theory: uncertainty associated with the probabilistic MC model (array of length N)
        sigma_meas: uncertainty assocated with the experimental measurements (array of length N)
        wavelength: independent variable (array of length N)
        l0: constant loss parameter (float)
        l1: effect of wavelength on losses (float)
    """
    
    loss = l0 + l1*rescale(wavelength)
    residual = (spect_data - (1-loss)*spect_theory)
    var_eff = sigma_theory**2 + sigma_data**2
    chi_square = np.sum(residual**2/var_eff)
    prefactor = 1/np.prod(np.sqrt(2*np.pi*var_eff))
    return prefactor * np.exp(-chi_square/2)

def log_posterior(theta, data, sample, phi_guess):
    """
    calculates log-posterior of a set of parameters producing an observed reflectance spectrum

    Parameters:
        theta: set of parameters (list-like)
        data: contains observed reflectance and uncertainties (Spectrum object)
        sample: contains information about the sample that produced data (Sample object) 
    """
    vol_frac, l0, l1 = theta
    theory_spectrum = do_some_monte_carlo_calculation(vol_frac, sample, data.wavelength)

    likelihood = calc_likelihood(theory_spectrum, data.meas, sigma_theory, data.sigma, data.wavelength, l0, l1)
    # Not sure where sigma_theory comes from here

    return np.log(likelihood) + np.log(calc_prior(theta, phi_guess))

def sample_parameters(data, sample, nwalkers=50, nsteps=500, burn_in_time=0, phi_guess = None):

    if phi_guess is None:
        phi_guess = 0.55 #This is a reasonable mid-point between dilute suspension and close-packed crystal

    # set expected values to initialize walkers
    expected_vals = np.array([phi_guess,.5,0])
    ndim = len(expected_vals)
    
    # set walkers in a distribution with width .05
    starting_positions = [expected_vals + 0.05*np.random.randn(ndim)) for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data, sample, phi_guess])
    sampler.run_mcmc(starting_positions, nsteps)
    traces = np.concatenate([sampler.chain[:,burn_in_time:,:], sampler.lnprobability[:,burn_in_time:,np.newaxis]],axis=2).reshape(-1, ndim+1).T
    return pd.DataFrame({key: traces[val] for val, key in enumerate(['vol_frac','l0','l1','lnprob'])})
