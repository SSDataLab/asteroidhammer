"""Functions to create and solve matrices for TESS data"""
from functools import lru_cache
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

import tess_ephem

from astropy.utils.data import download_file
from astropy.io import fits

import lightkurve as lk

from fbpca import pca


def _bin_down(dtime, dat, tpftime, dt=None, stddev=False):
    """Bins down high time resolutiqon files to the resolution of a TPF. """
    dtime, dat, tpftime = np.asarray(dtime, float), np.asarray(dat, float), np.asarray(tpftime, float)
    if dt is None:
        dt = 0.02083333 + np.zeros(len(tpftime) - 1)

    means = np.zeros((int(len(tpftime)), dat.shape[0]))
    if stddev:
        stds = np.zeros((int(len(tpftime)), dat.shape[0]))
    for idx in range(len(tpftime) - 1):
        mask = (dtime > tpftime[idx]) & (dtime <= (tpftime[idx] + dt[idx]))
        if mask.sum() == 0:
            continue
        for jdx in range(dat.shape[0]):
            means[idx, jdx] = np.mean(dat[jdx][mask])
            if stddev:
                stds[idx, jdx] = np.std(dat[jdx][mask])

    mask = (dtime > tpftime[idx]) & (dtime <= (tpftime[idx] + np.median(dt[idx])))
    if mask.sum() != 0:
        for jdx in range(dat.shape[0]):
            means[idx, jdx] = np.mean(dat[jdx][mask])
            if stddev:
                stds[idx, jdx] = np.std(dat[jdx][mask])
    if stddev:
        return np.vstack([np.vstack(means).T, np.vstack(stds).T])
    return np.vstack(means).T


@lru_cache()
def _download_quat(sector):
    """Download and cache the quaternions"""
    url = 'https://archive.stsci.edu/missions/tess/engineering/'
    df = pd.read_csv(url, skiprows=8, header=None)[:-2]
    df = df[['quat' in d[0] for idx, d in df.iterrows()]].reset_index(drop=True)
    df['url'] = [url + i.split('a href="')[1].split('">')[0] for i in df[0]]
    df['sector'] = [int(u.split('sector')[1].split('-quat.fits')[0]) for u in df['url']]
    path = download_file(df.loc[df.sector==sector, 'url'].iloc[0], cache=True, pkgname='asteroidhammer')
    return path


def quat_matrix(tpf=None, camera=None, sector=None, time=None, dt=None):
    """Get an array of the quaternions, at the time resolution of the input TPF"""
    if (tpf is None) and (camera is None) and (sector is None):
        raise ValueError('set either TPF or camera/sector')
    if camera is None:
        camera = tpf.camera
    if sector is None:
        sector = tpf.sector
    if time is None:
        time = np.asarray(tpf.time - tpf.hdu[1].data['TIMECORR'][tpf.quality_mask], float)

    hdu = fits.open(_download_quat(sector))
    dtime, dat = np.asarray(hdu[camera].data['time'], float), np.vstack([hdu[camera].data[f'C{camera}_Q{jdx + 1}'] for jdx in range(3)]).astype(float)
    return _bin_down(dtime, dat, time, dt=dt, stddev=True).T

@lru_cache()
def _download_cbv(sector, camera, ccd):
    """Download and cache the CBVs for a given sector, camera and CCD

    Note this function explicitly goes to the MAST archive and uses their file
    tree to find the best CBV. We go to the archive, and take the first file
    available in the FFI directory for a given sector. At time of writing, this is
    always the CBV, and let's us find new sector CBVs automatically.
    """
    url = 'https://archive.stsci.edu/missions/tess/ffi/' + 's{0:04}/'.format(sector)
    for count in [0, 1]:
        df = pd.read_csv(url, skiprows=8, header=None)[:-2]
        url += df[0][0].split('<a href="')[1].split('">')[0]
    url += f'{camera}-{ccd}/'
    df = pd.read_csv(url, skiprows=8, header=None)[:-2]
    url += df[0][0].split('<a href="')[1].split('">')[0]
    return url


def cbv_matrix(tpf=None, camera=None, sector=None, ccd=None, time=None, stddev=False, dt=None):
    """Returns the CBVs, binned to the same time array as the input TPF"""
    """Get an array of the quaternions, at the time resolution of the input TPF"""
    if (tpf is None) and (camera is None) and (sector is None) and (ccd is None):
        raise ValueError('set either TPF or camera/sector/ccd')
    if camera is None:
        camera = tpf.camera
    if sector is None:
        sector = tpf.sector
    if ccd is None:
        ccd = tpf.ccd
    if time is None:
        time = np.asarray(tpf.time - tpf.hdu[1].data['TIMECORR'][tpf.quality_mask], float)

    url = _download_cbv(sector, camera, ccd)
    path = download_file(url, cache=True, pkgname='asteroidhammer')
    hdu = fits.open(path)

    cbvtime = np.asarray(hdu[2].data['TIME'], float)
    cbvs = np.hstack([np.vstack([hdu[jdx].data[f'VECTOR_{idx + 1}'] for idx in range(2)]).T for jdx in np.arange(2, 6)])
    return _bin_down(cbvtime, cbvs.T, time, stddev=stddev, dt=dt).T


def poly_matrix(time, npoly=3):
    """Returns a matrix of polynomial """
    # Time polynomial
    t = (time - time.mean())
    t /= (t.max() - t.min())
    poly = np.vstack([t**idx for idx in np.arange(0, npoly + 1)]).T
    return poly


def spline_matrix(time, degree=3, n_knots=10):
    """Returns a spline design matrix"""
    knots = np.linspace(time[0], time[-1], n_knots + 2)[1:-1]
    return lk.designmatrix.create_spline_matrix(time, degree=3, knots=list(knots)).X


def pca_matrix(time, flux, pix_mask, n_pca_components=3, tmask=None):
    """Finds the PCA components for pixels in the TPF

    Set `percentile` to a value between 0 and 1. Only pixels below
    this percentile will be used. """
#    for percentile in [faint_percentile, bright_percentile]:
#        if (percentile <= 0) | (percentile > 100):
#            raise ValueError("`percentile` must be between 0 and 1")
#    faint = np.nanmedian(flux, axis=0) < np.nanpercentile(np.median(flux, axis=0), faint_percentile)
#    bright = np.nanmedian(flux, axis=0) < np.nanpercentile(np.median(flux, axis=0), bright_percentile)

    U, s, V = pca(flux[:, pix_mask], k=n_pca_components, n_iter=10)
    return U
    #
    # A = np.hstack([U, poly_matrix(time, 3)])
    #
    # mod = np.zeros_like(flux[:, bright_mask])
    # if tmask is None:
    #     tmask = np.ones(len(flux), bool)
    # Am = A[tmask]
    # sigma_w_inv = Am.T.dot(Am)
    # for idx, f in zip(range(bright_mask.sum()), flux[tmask][:, bright_mask].T):
    #     B = Am.T.dot(f)
    #     w = np.linalg.solve(sigma_w_inv, B)
    #     mod[:, idx] = A.dot(w)
    # U2 = pca((flux[:, bright_mask] - mod), n_pca_components)[0]
    # return np.hstack([U, U2])
#
# #@lru_cache()
# def get_straps(ar, ar_e, pix_mask):
#     straps = np.average(np.nan_to_num(ar), axis=1,
#                         weights=np.nan_to_num(((np.nanstd(ar, axis=0) < 5) & pix_mask).astype(float)/ar_e+ 1e-10) + 1e-10)
#     return straps[:, None, :] * np.ones_like(ar)
#
# #@lru_cache()
# def get_smooth_frame_correction(ar, pix_mask, npoly=3):
#     mask = ((np.nanstd(ar, axis=0) < 5) & np.copy(pix_mask) & np.isfinite(ar))
#     X, Y = np.mgrid[:ar.shape[1], :ar.shape[2]]
#     X, Y = ((X - X.mean())/(X.max() - X.min())).ravel(), ((Y - Y.mean())/(Y.max() - Y.min())).ravel()
#
#     X1 = np.vstack([X**idx for idx in range(npoly + 1)]).T
#     Y1 = np.vstack([Y**idx for idx in range(npoly + 1)]).T
#     poly = np.hstack([X1 * Y1[:, idx][:, None] for idx in range(npoly + 1)])
#     poly_mod = np.zeros_like(ar)
#
#     for tdx in range(ar.shape[0]):
#         y = ar[tdx].ravel()
#         k = mask[tdx].ravel()
#         if not k.any():
#             continue
#         sigma_w_inv = poly[k].T.dot(poly[k])
#         B = poly[k].T.dot(y[k])
#         try:
#             w = np.linalg.solve(sigma_w_inv, B)
#         except:
#             continue
#         mod = poly.dot(w)
#         res = y - mod
#         percs = np.nanpercentile((y - mod)[k], [5, 95])
#         k &= (res > percs[0]) & (res < percs[1])
#
#         sigma_w_inv = poly[k].T.dot(poly[k])
#         B = poly[k].T.dot(y[k])
#         try:
#             w = np.linalg.solve(sigma_w_inv, B)
#         except:
#             continue
#         mod = poly.dot(w)
#         poly_mod[tdx] = mod.reshape(ar.shape[1:])
#
#     return poly_mod

#
# def multiply_self(matrix):
#     """Take the product of a matrix with itself"""
#     return np.vstack([np.vstack([matrix[idx][None, :] * np.atleast_2d(matrix[idx + 1:]) for idx in np.arange(0, matrix.shape[0] - 1), matrix])])


def linear_solve(A, y, ye=None, tmask=None, fullmask=None, prior_mu=None, prior_sigma=None):
    """Linearly solve an image stack..."""

    mod = np.ones_like(y)
    if prior_sigma is None:
        prior_sigma = np.ones(A.shape[1]) * 1e10

    if prior_mu is None:
        prior_mu = np.zeros(A.shape[1])

    if (ye is None) and (fullmask is None):
        # simple case, this is the fastest to compute
        if tmask is None:
            As = A.copy()
        else:
            As = A[tmask]
        AsT = As.T
        sigma_w_inv = AsT.dot(As)
        sigma_w_inv += np.diag(1/prior_sigma**2)
        if tmask is None:
            ys = np.copy(ys)
        else:
            ys = np.copy(y[tmask])
        for idx in range(ys.shape[1]):
            for jdx in range(ys.shape[2]):
                B = AsT.dot(ys[:, idx, jdx])
                B += prior_mu/prior_sigma**2
                w = np.linalg.solve(sigma_w_inv, B)
                mod[:, idx, jdx] = A.dot(w)

    # The following cases are slower to compute
    elif (fullmask is None) and (ye is not None):
        AT = A.T
        for idx in range(y.shape[1]):
            for jdx in range(y.shape[2]):
                sigma_w_inv = AT.dot(A/ye[:, idx, jdx][:, None]**2)
                sigma_w_inv += np.diag(1/prior_sigma**2)
                B = AT.dot(y[:, idx, jdx]/ye[:, idx, jdx]**2)
                B += prior_mu/prior_sigma**2
                w = np.linalg.solve(sigma_w_inv, B)
                mod[:, idx, jdx] = A.dot(w)

    elif (ye is None) and (fullmask is not None):
        for idx in range(y.shape[1]):
            for jdx in range(y.shape[2]):
                mask = fullmask[:, idx, jdx]
                if not mask.any():
                    continue
                sigma_w_inv = A[mask].T.dot(A[mask])
                sigma_w_inv += np.diag(1/prior_sigma**2)
                B = A[mask].T.dot(y[mask, idx, jdx])
                B += prior_mu/prior_sigma**2
                w = np.linalg.solve(sigma_w_inv, B)
                mod[:, idx, jdx] = A.dot(w)
    else:
        for idx in range(y.shape[1]):
            for jdx in range(y.shape[2]):
                mask = fullmask[:, idx, jdx]
                if not mask.any():
                    continue
                sigma_w_inv = A[mask].T.dot(A[mask]/ye[mask, idx, jdx][:, None]**2)
                sigma_w_inv += np.diag(1/prior_sigma**2)
                B = A[mask].T.dot(y[mask, idx, jdx]/ye[mask, idx, jdx]**2)
                B += prior_mu/prior_sigma**2
                w = np.linalg.solve(sigma_w_inv, B)
                mod[:, idx, jdx] = A.dot(w)
    return mod
