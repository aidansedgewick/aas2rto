from logging import getLogger

import numpy as np

from scipy.optimize import curve_fit

from astropy.time import Time

import emcee

from dk154_targets.target import Target

from dk154_targets import paths

logger = getLogger("bazin_model")

DEFAULT_PARAM_NAMES = ("A", "t0", "t_fall", "t_rise", "c")


def bazin_flux(t, A, t0, t_fall, t_rise, c):
    numer = np.exp((t - t0) / t_fall)
    denom = 1.0 + np.exp((t - t0) / t_rise)

    return A * numer / denom + c


def bazin_log_prior(params):
    if any(params < 0):
        return -np.inf

    if any(params > 1e6):
        return -np.inf
    return 0.0


def bazin_log_likelihood(params, t, fl_obs, fl_err):
    fl_model = bazin_flux(t, *params)
    sigma2 = fl_err**2

    fl_diff = fl_obs - fl_model
    ll = -0.5 * np.sum(fl_diff**2 / sigma2)
    if not np.isfinite(ll):
        print(params)
        print(fl_model)
        print(fl_obs)
        return -np.inf
    return ll


def bazin_log_probability(params, t, fl_obs, fl_err):
    lp = bazin_log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + bazin_log_likelihood(params, t, fl_obs, fl_err)


def guess_p0(t_shift, fl_obs):

    fl_argmax = np.argmax(fl_obs)
    t0 = t_shift[fl_argmax]
    A = fl_obs[fl_argmax]

    return A, t0, 19.0, 19.0, 0.0


class BazinModel:

    def __init__(self, params, result: dict = None, p_err=None, per_band=False):
        self.params = params
        self.p_err = p_err
        self.per_band = per_band
        self.result = result

        self.vparam_names = DEFAULT_PARAM_NAMES
        if result is not None:
            self.vparam_names = self.result.get("vparam_names")

    def bandflux(self, band, t, zp=8.9, zpsys="ab"):

        if self.per_band:
            raise NotImplementedError(
                "must use per_band=False for now... (one set of params for whole LC...)"
            )
            if band in self.params:
                band_params = self.params[band]
                param_arr = tuple([band_params[k] for k in self.vparam_names])
        else:
            param_arr = tuple([self.params[k] for k in self.vparam_names])

        return bazin_flux(t, *param_arr)

    def bandmag(self, band, t, zp=8.9, zpsys="ab"):
        fl = self.bandflux(band, t)

        if zpsys.lower() != "ab":
            raise ValueError(f"no sys {zpsys}: only 'ab'")
        return -2.5 * np.log10(fl) + zp

    def update(self, new_params):
        self.params.update(new_params)


def fit_bazin(
    t_obs,
    mag,
    magerr,
    use_emcee=False,
    nwalkers=12,
    nsteps=5000,
    burnin=500,
    vparam_names=DEFAULT_PARAM_NAMES,
):

    fl_obs = 10 ** (-0.4 * (mag - 8.9))
    fl_err = 1.09 / magerr

    t_offset = t_obs[0]
    t_shift = t_obs - t_offset

    p0_guess = guess_p0(t_shift, fl_obs)
    try:
        lsq_params, pcov = curve_fit(
            bazin_flux, t_shift, fl_obs, p0=p0_guess, sigma=fl_err, maxfev=100000
        )
    except Exception as e:
        print(e)
        return None, None

    if use_emcee:
        ndim = len(p0_guess)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            bazin_log_probability,
            args=(t_shift, fl_obs, fl_err),
        )

        # nudge = lsq_params * 1e-3 * np.random.randn(nwalkers, ndim)
        # pos = p0_guess + nudge
        # print(pos)
        pos = np.column_stack(
            [
                np.random.normal(x, s, nwalkers)
                for x, s in zip(lsq_params, [1e-5, 1.0, 1.0, 1.0, 1e-5])
            ]
        )
        sampler.run_mcmc(pos, nsteps)

        samples = sampler.get_chain(discard=burnin, flat=True)

        samples[:, 1] = samples[:, 1] + t_offset

        # samples = np.column_stack(
        #     [
        #         np.random.normal(x, s, nwalkers * nsteps)
        #         for x, s in zip(lsq_params, [1e-5, 1.0, 1.0, 1.0, 1e-5])
        #     ]
        # )
        param_arr = np.mean(samples, axis=0)
        result = dict(samples=samples, lsq_pcov=pcov, vparam_names=vparam_names)

    else:
        param_arr = lsq_params
        param_arr[1] = param_arr[1] + t_offset
        result = dict(lsq_pcov=pcov, vparam_names=vparam_names)
    params = {k: v for k, v in zip(vparam_names, param_arr)}
    return params, result


class FitBazinModel:

    default_band_col = "band"
    default_date_col = "mjd"
    default_mag_col = "mag"
    default_magerr_col = "magerr"

    __name__ = "bazin"

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
        self.burnin = burnin

    def __call__(self, target: Target, t_ref: Time):
        objectId = target.objectId
        lightcurve = target.compiled_lightcurve

        if lightcurve is None:
            logger.info(f"{objectId} has no compiled lightcurve.")
            return None

        if self.per_band:
            raise NotImplementedError(
                "must use per_band=False for now... (one set of params for whole LC...)"
            )

            params = {}
            result = {}
            for ii, (band, band_lc) in enumerate(lightcurve.groupby(self.band_col)):
                t_dat = band_lc[self.date_col].values
                mag_dat = band_lc[self.mag_col].values
                magerr_dat = band_lc[self.magerr_col].values

                params_ii, result_ii = fit_bazin(
                    t_dat,
                    mag_dat,
                    magerr_dat,
                    self.use_emcee,
                    nwalkers=self.nwalkers,
                    nsteps=self.nsteps,
                    burnin=self.burnin,
                )
                params[band] = params_ii
                result[band] = result_ii

        else:
            t_dat = lightcurve[self.date_col].values
            mag_dat = lightcurve[self.mag_col].values
            magerr_dat = lightcurve[self.magerr_col].values

            params, result = fit_bazin(
                t_dat,
                mag_dat,
                magerr_dat,
                self.use_emcee,
                nwalkers=self.nwalkers,
                nsteps=self.nsteps,
                burnin=self.burnin,
            )
        return BazinModel(params, result=result, per_band=self.per_band)
