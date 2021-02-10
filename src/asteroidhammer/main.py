import numpy as np
from .matrices import *
from .query import *

from scipy import sparse

from astropy.stats import sigma_clip

class Data(object):
    def __init__(self, time, flux, error, camera, sector, ccd, dt=None, xmin=None, xmax=None, ymin=None, ymax=None):
        self.time = time
        self.raw_flux = flux
        self.error = error
        self.camera = camera
        self.sector = sector
        self.ccd = ccd
        self.dt = dt
        self.shape = self.raw_flux.shape

        if xmin is None:
            self.xmin = 0
        else:
            self.xmin = xmin
        if ymin is None:
            self.ymin = 0
        else:
            self.ymin = ymin
        if xmax is None:
            self.xmax = self.shape[1]
        else:
            self.xmax = xmax
        if ymax is None:
            self.ymax = self.shape[2]
        else:
            self.ymax = ymax

        self.Y, self.X = np.mgrid[self.ymin:self.ymax, self.xmin:self.xmax]
        med = np.median(self.raw_flux, axis=0)
        self.bright_mask = (med - np.median(med)) > 5

        # Find where the straps are on the detector
        strap_line = ((self.raw_flux - med) * (~self.bright_mask)).sum(axis=(0, 1))/(~self.bright_mask).sum(axis=0)
        self.strap_mask = sigma_clip(strap_line).mask | sigma_clip(np.gradient(strap_line)).mask
        self.strap_mask = (np.gradient(self.strap_mask.astype(float)) != 0) | self.strap_mask

        self.flux = np.copy(flux) - self._basic_bkg()

        self.downlink = np.where(np.diff(self.time)/np.median(self.dt) > 20)[0][0]

        self.quats = quat_matrix(camera=camera, sector=sector, time=time, dt=dt)
        self.cbvs = cbv_matrix(camera=camera, sector=sector, ccd=ccd, time=time, dt=dt, stddev=False)
        self.poly = poly_matrix(time=time, npoly=3)
        self.spline = spline_matrix(time, n_knots=10)

        # Find bad times
        self.bad_times = self._find_bad_frames_based_on_grad()
        mask = np.asarray([sigma_clip(np.gradient(np.ma.masked_array(self.quats[:, idx], self.bad_times)), sigma=6).mask for idx in range(self.quats.shape[1])])
        self.bad_times |= mask.sum(axis=0) >= 3
        mask = np.asarray([sigma_clip(np.gradient(np.ma.masked_array(self.cbvs[:, idx], self.bad_times)), sigma=6).mask for idx in range(self.cbvs.shape[1])])
        self.bad_times |= mask.sum(axis=0) >= 3


        self.pca1 = pca_matrix(time=time[:self.downlink], flux=flux[:self.downlink],
                               pix_mask=~self.bright_mask,
                               n_pca_components=3, tmask=~self.bad_times[:self.downlink])
        self.pca2 = pca_matrix(time=time[self.downlink:], flux=flux[self.downlink:],
                               pix_mask=~self.bright_mask,
                               n_pca_components=3, tmask=~self.bad_times[self.downlink:])



    def __repr__(self):
        return (f'{self.shape} Sector {self.sector}, Camera {self.camera}, CCD {self.ccd}')

    @staticmethod
    def from_tpf(self):
        time = np.asarray(tpf.time, float)
        flux, error = tpf.flux, tpf.flux_err
        camera, sector, ccd = tpf.camera, tpf.sector, tpf.ccd
        timecorr = tpf.hdu[1].data['TIMECORR'][tpf.quality_mask]
        dt = (time - timecorr)[1:] - (time - timecorr)[:-1]
        return Data(time=time, flux=time, error=error, camera=camera, sector=sector, ccd=ccd, dt=dt)


    def _find_bad_frames_based_on_grad(self):
        """ Finds bad frames based on gradient of flux"""
        dif = np.gradient(self.flux, self.time, axis=(0))
        # 5th and 95th percentile of gradient
        perc = np.percentile(dif, [5, 95], axis=0)
        # Difference between 5th and 95th percentile
        perc = np.diff(perc, axis=0)[0]
        # Remove pixels that have a large 95th-5th percentile
        bad_pix = perc > np.nanpercentile(perc, 99)
        std = dif[:, ~bad_pix].std()
        # Find times that are 2sigma outliers based on all pixels, for a significant fraction of time.
        bad_times = ((np.abs(dif) * (~bad_pix).astype(float) > (2 * std))).sum(axis=(1, 2))
        bad_times = bad_times/np.product(dif.shape[1:])
        return np.in1d(np.arange(self.shape[0]), np.where(bad_times > 0.01)[0] + 1)

    def _basic_bkg(self):
        x = ((self.X - self.X.mean())/(self.X.max() - self.X.min()))
        y = ((self.Y - self.Y.mean())/(self.Y.max() - self.Y.min()))
        lines = np.asarray([(x == x1).astype(float).ravel() for x1 in x[0][self.strap_mask]]).T

        x = np.hstack([x.ravel()[:, None]**idx for idx in range(5)])
        y = np.hstack([y.ravel()[:, None]**idx for idx in range(5)])
        A = np.hstack([x * y1[:, None] for y1 in y.T])
        f = np.copy(self.raw_flux)
        fe = np.copy(self.error)
        model = np.zeros_like(f)
        pix_mask = (~self.bright_mask).ravel()

        prior_sigma = np.ones(A.shape[1]) * 5000
        prior_mu = np.zeros(A.shape[1])

        prior_sigma = np.hstack([prior_sigma, np.ones(lines.shape[1]) * 500])
        prior_mu = np.hstack([prior_mu, np.zeros(lines.shape[1])])
        A = np.hstack([A, lines])

        for idx in tqdm(range(self.shape[0]), desc='Basic Scattered Light Correction'):
            A1 = A[pix_mask]
            sigma_w_inv = A1.T.dot(A1/fe[idx].ravel()[pix_mask][:, None]**2)
            sigma_w_inv += np.diag(1/prior_sigma**2)
            B = A1.T.dot(f[idx].ravel()[pix_mask]/fe[idx].ravel()[pix_mask]**2)
            B += prior_mu/prior_sigma**2
            mod = A.dot(np.linalg.solve(sigma_w_inv, B))

            #clip outliers
            res = f[idx].ravel() - mod
            k = res < (3 * _std_iter(res, mask=pix_mask))
            A1 = A[pix_mask & k]
            sigma_w_inv = A1.T.dot(A1/fe[idx].ravel()[pix_mask & k][:, None]**2)
            sigma_w_inv += np.diag(1/prior_sigma**2)
            B = A1.T.dot(f[idx].ravel()[pix_mask & k]/fe[idx].ravel()[pix_mask & k]**2)
            B += prior_mu/prior_sigma**2
            model[idx] = A.dot(np.linalg.solve(sigma_w_inv, B)).reshape(self.shape[1:])

        return model


def _std_iter(x, mask, sigma=3, n_iters=3):
    """ Iteratively finds the standard deviation of an array after sigma clipping

    Parameters
    ----------
    x : np.ndarray
        Array with average of zero
    mask : np.ndarray of bool
        Mask of same size as x, where True indicates a point to be masked.
    sigma : int or float
        The standard deviation at which to clip
    n_iters : int
        Number of iterations
    """
    m = mask.copy()
    for iter in range(n_iters):
        std = np.std(x[~m])
        m |= (np.abs(x) > (std * sigma))
    return std
