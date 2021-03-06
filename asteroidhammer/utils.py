"""Functions useful for asteroidhammer"""
from functools import lru_cache
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patheffects as path_effects
from tqdm import tqdm

import tess_ephem

from astropy.utils.data import download_file
from astropy.io import fits

import lightkurve as lk

from fbpca import pca

def two_panel_movie(data_a, data_b, out='out.mp4', scale='linear',
                    title_a='', title_b='', **kwargs):
    '''Create an mp4 movie of a 3D array
    '''
    if scale == 'log':
        data_A = np.log10(np.copy(data_a), stack)
        data_B = np.log10(np.copy(data_b), stack)
    else:
        data_A = np.copy(data_a)
        data_B = np.copy(data_b)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax in axs:
        ax.set_facecolor('#ecf0f1')
    im1 = axs[0].imshow(data_A[0], origin='bottom', **kwargs)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title(title_a, fontsize=10)
    im2 = axs[1].imshow(data_B[0], origin='bottom', ** kwargs)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title(title_b, fontsize=10)
    def animate(i):
        im1.set_array(data_A[i])
        im2.set_array(data_B[i])
    anim = animation.FuncAnimation(fig, animate, frames=len(data_A), interval=30)
    anim.save(out, dpi=150)
    del data_A, data_B


def movie(data, asteroids={}, out='out.mp4', scale='linear',
                    title='', **kwargs):
    # This makes things faster
    if isinstance(data, lk.targetpixelfile.TargetPixelFile):
        shape = data.shape
        data = np.copy(data.flux)
    elif isinstance(data, np.ndarray):
        shape = data.shape
    else:
        raise ValueError('can not parse input')

    if len(asteroids) != 0:
        for label in ['column', 'row', 'names']:
            if label not in asteroids:
                raise ValueError('can not parse `asteroids`')
            if len(asteroids[label]) != len(data):
                if label is 'names':
                    continue
                raise ValueError(f'asteroid {label} input is not same length as data')
        c = asteroids['column']
        r = asteroids['row']
        names = asteroids['names']

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.set_facecolor('#ecf0f1')
    im = ax.imshow(data[0], origin='bottom', **kwargs)
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    if len(asteroids) != 0:
        scat = ax.scatter(c[0], r[0], facecolor='None', s=100, edgecolor='r')
        scat.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                               path_effects.Normal()])
        texts = [ax.text(-10, -10, '', color='red', size=6, ha='left', va='center') for idx in range(c.shape[1])]
        [text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                               path_effects.Normal()]) for text in texts]

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_xticks([])
    ax.set_yticks([])
    def animate(i):
        im.set_array(data[i])
        if len(asteroids) != 0 :
            scat.set_offsets(np.vstack([c[i], r[i]]).T)
            for jdx in range(c.shape[1]):
                coords = c[i][jdx], r[i][jdx]
                if ((coords[0] > 10) & (coords[0] < (shape[1] - 10)) &
                    (coords[1] > 10) & (coords[1] < (shape[2] - 10))):
                    texts[jdx].set_x(coords[0])
                    texts[jdx].set_y(coords[1])
                    texts[jdx].set_text(names[jdx])
                else:
                    texts[jdx].set_x(-10)
                    texts[jdx].set_y(-10)
                    texts[jdx].set_text('')

        if len(asteroids) != 0:
            return im, scat, texts,
        return im
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=30)
    anim.save(out, dpi=150)


def get_asteroid_dict(tpf, max_objs=None, magnitude_lower_limit=100, magnitude_upper_limit=0):
    mask = np.in1d(np.arange(len(tpf)), np.arange(0, len(tpf) * 4, 4))
    objs = tpf.query_solar_system_objects(cadence_mask=mask)
    objs = objs.drop_duplicates("Name")
    objs = objs[(objs.Mv < magnitude_lower_limit) & (objs.Mv > magnitude_upper_limit)]
    objs = objs.sort_values('Mv').reset_index(drop=True)
    if max_objs is not None:
        objs = objs.head(max_objs)

    names = np.asarray(objs.sort_values('Mv').Name)
    ephs = []
    for name in tqdm(names):
        eph = tess_ephem.ephem(name, tpf.astropy_time)
        eph = eph[(eph.sector == tpf.sector) & (eph.camera == tpf.camera) & (eph.ccd == tpf.ccd)]
        ephs.append(eph)

    c = np.zeros((len(tpf), (len(ephs)))) * np.nan
    r = np.zeros((len(tpf), (len(ephs)))) * np.nan
    for jdx, eph in enumerate(ephs):
        c[np.in1d(tpf.astropy_time.jd, np.asarray([i.jd for i in eph.index])), jdx] = np.asarray(eph.column)
        r[np.in1d(tpf.astropy_time.jd, np.asarray([i.jd for i in eph.index])), jdx] = np.asarray(eph.row)

    c -= tpf.column
    r -= tpf.row
    return {'time':tpf.astropy_time.jd, 'column':c, 'row':r, 'names':names, 'Mv':np.asarray(objs['Mv'])}


def bin_down(dtime, dat, tpftime, dt=None, stddev=False):
    """Bins down high time resolution files to the resolution of a TPF. """
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


def get_quats(tpf=None, camera=None, sector=None, time=None, dt=None):
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
    # dt = np.median(np.diff(time))
    # def run(dtime, dat, tpftime):
    #     means = np.zeros((int(len(tpftime)), 3))
    #     stds = np.zeros((int(len(tpftime)), 3))
    #     for idx in range(len(tpftime) - 1):
    #         mask = (dtime > tpftime[idx]) & (dtime <= (tpftime[idx] + dt))
    #         if mask.sum() == 0:
    #             continue
    #         for jdx in range(3):
    #             means[idx, jdx] = np.mean(dat[jdx][mask])
    #             stds[idx, jdx] = np.std(dat[jdx][mask])
    #
    #     mask = (dtime > tpftime[idx])
    #     if mask.sum() != 0:
    #         for jdx in range(3):
    #             means[idx, jdx] = np.mean(dat[jdx][mask])
    #             stds[idx, jdx] = np.std(dat[jdx][mask])
    #     return np.vstack([np.vstack(means).T, np.vstack(stds).T])
    return bin_down(dtime, dat, time, dt=dt, stddev=True).T

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


def get_cbvs(tpf=None, camera=None, sector=None, ccd=None, time=None, stddev=False, dt=None):
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
    return bin_down(cbvtime, cbvs.T, time, stddev=stddev, dt=dt).T


def get_poly(time, npoly=3):
    """Returns a matrix of polynomial """
    # Time polynomial
    t = (time - time.mean())
    t /= (t.max() - t.min())
    poly = np.vstack([t**idx for idx in np.arange(0, npoly + 1)]).T
    return poly


@lru_cache()
def get_spline(tpf, degree=3, n_knots=10):
    """Returns a spline design matrix"""
    knots = np.linspace(tpf.time[0], tpf.time[-1], n_knots + 2)[1:-1]
    return lk.designmatrix.create_spline_matrix(tpf.time, degree=3, knots=list(knots)).X



def get_pca(time, flux, faint_percentile=20, bright_percentile=80, n_pca_components=3, tmask=None):
    """Finds the PCA components for pixels in the TPF

    Set `percentile` to a value between 0 and 1. Only pixels below
    this percentile will be used. """
    for percentile in [faint_percentile, bright_percentile]:
        if (percentile <= 0) | (percentile > 100):
            raise ValueError("`percentile` must be between 0 and 1")
    faint = np.nanmedian(flux, axis=0) < np.nanpercentile(np.median(flux, axis=0), faint_percentile)
    bright = np.nanmedian(flux, axis=0) < np.nanpercentile(np.median(flux, axis=0), bright_percentile)

    U, s, V = pca(flux[:, faint], k=n_pca_components, n_iter=10)

    A = np.hstack([U, get_poly(time, 3)])


    mod = np.zeros_like(flux[:, bright])
    if tmask is None:
        tmask = np.ones(len(flux), bool)
    Am = A[tmask]
    sigma_w_inv = Am.T.dot(Am)
    for idx, f in zip(range(bright.sum()), flux[tmask][:, bright].T):
        B = Am.T.dot(f)
        w = np.linalg.solve(sigma_w_inv, B)
        mod[:, idx] = A.dot(w)
    U2 = pca((flux[:, bright] - mod), n_pca_components)[0]
    return np.hstack([U, U2])

#@lru_cache()
def get_straps(ar, ar_e, pix_mask):
    straps = np.average(np.nan_to_num(ar), axis=1,
                        weights=np.nan_to_num(((np.nanstd(ar, axis=0) < 5) & pix_mask).astype(float)/ar_e+ 1e-10) + 1e-10)
    return straps[:, None, :] * np.ones_like(ar)

#@lru_cache()
def get_smooth_frame_correction(ar, pix_mask, npoly=3):
    mask = ((np.nanstd(ar, axis=0) < 5) & np.copy(pix_mask) & np.isfinite(ar))
    X, Y = np.mgrid[:ar.shape[1], :ar.shape[2]]
    X, Y = ((X - X.mean())/(X.max() - X.min())).ravel(), ((Y - Y.mean())/(Y.max() - Y.min())).ravel()

    X1 = np.vstack([X**idx for idx in range(npoly + 1)]).T
    Y1 = np.vstack([Y**idx for idx in range(npoly + 1)]).T
    poly = np.hstack([X1 * Y1[:, idx][:, None] for idx in range(npoly + 1)])
    poly_mod = np.zeros_like(ar)

    for tdx in range(ar.shape[0]):
        y = ar[tdx].ravel()
        k = mask[tdx].ravel()
        if not k.any():
            continue
        sigma_w_inv = poly[k].T.dot(poly[k])
        B = poly[k].T.dot(y[k])
        try:
            w = np.linalg.solve(sigma_w_inv, B)
        except:
            continue
        mod = poly.dot(w)
        res = y - mod
        percs = np.nanpercentile((y - mod)[k], [5, 95])
        k &= (res > percs[0]) & (res < percs[1])

        sigma_w_inv = poly[k].T.dot(poly[k])
        B = poly[k].T.dot(y[k])
        try:
            w = np.linalg.solve(sigma_w_inv, B)
        except:
            continue
        mod = poly.dot(w)
        poly_mod[tdx] = mod.reshape(ar.shape[1:])

    return poly_mod


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
