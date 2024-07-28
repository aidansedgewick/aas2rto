from logging import getLogger

import numpy as np

from scipy.optimize import curve_fit

import emcee

from dk154_targets.target import Target

from dk154_targets import paths

logger = getLogger("bazin_model")


def bazin_flux(t, A, t0, t_fall, t_rise, c):
    numer = np.exp((t - t0) / t_fall)
    denom = 1.0 + np.exp((t - t0) / t_rise)

    return A * numer / denom + c


def bazin_log_prior(params):
    if params[2] < params[3]:
        return -np.inf
    return 0.0


def bazin_log_likelihood(params, t, fl_obs, fl_err):
    fl_model = bazin_flux(t, *params)
    sigma2 = fl_err**2

    fl_diff = fl_obs - fl_model

    return -0.5 * np.sum(fl_diff**2 / sigma2)


def bazin_log_probability(params, t, fl_obs, fl_err):
    lp = bazin_log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + bazin_log_likelihood(params, t, fl_obs, fl_err)


class BazinModel:

    def __init__(self, params, p_err=None, per_band=False):
        self.params = params
        self.p_err = p_err
        self.per_band = per_band

    def bandflux(self, band, t):

        if self.per_band:
            if band in self.params:
                band_params = self.params[band]
        else:
            band_params = self.params

        return bazin_flux(t, *band_params)

    def bandmag(self, band, sys, t):
        fl = self.bandflux(band, t)

        if sys.lower() != "ab":
            raise ValueError(f"no sys {sys}: only 'ab'")
        return -2.5 * np.log10(fl) + 8.9


def guess_p0(t_obs, fl_obs):

    fl_argmax = np.argmax(fl_obs)
    t0 = t_obs[fl_argmax]
    A = fl_obs[fl_argmax]

    return A, t0, 19.0, 19.0, 0.0

class BazinModelResult:
    def __init__(self, samples=None):
        self.samples = samples



def fit_bazin(t_obs, mag, magerr, use_emcee=False, nwalkers=12, nsteps=5000, burnin=500, ):

        fl_obs = 10 ** (-0.4 * (mag - 8.9))
        fl_err = 1.09 / magerr

        p0_guess = guess_p0(t_obs, fl_obs)
        try:
            lsq_params, pcov = curve_fit(
                bazin_flux, t_obs, fl_obs, p0=p0_guess, yerr=fl_err
            )
        except Exception as e:
            return None

        if use_emcee:
            ndim = len(p0_guess)
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                bazin_log_probability,
                args=(t_obs, fl_obs, fl_err),
            )
            sampler.run_mcmc(lsq_params, nsteps)

            samples = sampler.get_chain(discard=burnin, flat=True)
            params = np.mean(samples, axis=0)

            

        else:
            return lsq_params, BazinModelResult()

class FitBazinModel:

    default_band_col = "band"
    default_date_col = "mjd"
    default_mag_col = "mag"
    default_magerr_col = "magerr"

    def __init__(
        self,
        per_band=False,
        use_emcee=False,
        band_col=None,
        date_col=None,
        mag_col=None,
        magerr_col=None,
        nwalkers=12,
        nsteps=5000,
        burnin=500,
    ):
        self.per_band = per_band

        self.use_emcee = use_emcee

        self.band_col = band_col or self.default_band_col
        self.date_col = date_col or self.default_date_col
        self.mag_col = mag_col or self.default_mag_col
        self.magerr_col = magerr_col or self.default_magerr_col

        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnin = self.burnin

    def __call__(self, target: Target):

        objectId = target.objectId
        lightcurve = target.compiled_lightcurve

        if lightcurve is None:
            logger.info(f"{objectId} ")
            return None

        if self.per_band:
            
            params = {}
            for band, band_lc in lightcurve.groupby(self.band_col):



        else:
            
