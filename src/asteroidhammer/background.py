'''tools to remove the background from TESS FFIs in a fast, light weight way'''
import fitsio
import numpy as np
from tqdm import tqdm
from scipy import sparse

import lightkurve as lk

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time

import matplotlib.pyplot as plt

from .query import query_gaia_sources


def _count_unq(locs):
    """Finds unique tuples"""
    s = np.asarray([f'{l[0]},{l[1]}' for l in locs.T])
    i, j, k = np.unique(s, return_counts=True, return_inverse=True)
    return k[j]

## Array for circular apertures as a function of g magnitude...
_christina_guess = np.asarray([[0, 10], [1, 10], [2, 10], [3, 9], [4, 7], [5, 5], [6, 4], [8, 4], [10, 4], [11, 3], [12, 3], [13, 3], [14, 2], [15, 2], [16, 2], [17, 1]])

class BackgroundCorrector(object):


    def __init__(self, fnames, nchunks=16, testframe=None, magnitude_limit=16):
        """Parameters

        fnames : list of str
            List containing paths to the FFI files
        nchunks : int
            Number of chunks to split the data into when querying Gaia
        """

        self.nfiles = len(fnames)
        self.nchunks = int(nchunks)
        self.npixels = 2048//self.nchunks
        self.fnames = fnames
        self.magnitude_limit = magnitude_limit

        if testframe is None:
            testframe = len(fnames)//2
        self.testframe = testframe
        self.wcs = WCS(fits.open(fnames[self.testframe])[1])

        row, column = np.mgrid[:2078, :2136]
        self.column, self.row = column[:2048, 45:2048+45], row[:2048, 45:2048+45]
        ra, dec = self.wcs.all_pix2world(self.column.ravel(), self.row.ravel(), 0)
        ra, dec = ra.reshape(self.column.shape), dec.reshape(self.column.shape)
        self.locs, self.dra, self.ddec, self.phot = self._get_star_aperture(self.fnames[self.testframe], magnitude_limit=self.magnitude_limit)

        # Christina's empirical calibration
        gaia_zp = 25.688365751
        self.mag = gaia_zp + -2.5*np.log10(self.phot)
        count = _count_unq(self.locs)

        self.star_aper = np.zeros((2078, 2136), bool)
        self.star_aper[self.locs[0], self.locs[1] + 45] = True
        self.mask = (self.mag < 13) & (count == 1) & (np.hypot(self.dra, self.ddec) <= 3 * np.median(np.hypot(np.diff(ra[0]), np.diff(dec[:, 0]))))
        f = fitsio.read(fnames[self.testframe], ext=1)
        bkg_aper = ~self.star_aper
        bkg_aper[f < 10] = False

        # For testing
        bkg_aper[self.npixels:] = False

        bkg_aper &= f < np.nanpercentile(f[bkg_aper], 99.5)
        bkg_aper &= f > np.nanpercentile(f[bkg_aper], 0.05)
        self.bkg_aper = bkg_aper

        # Row short for test!
        self.Xf = self._get_X(c=column[0, :2048], r=row[:128, 0])

        # Clip out outliers present in the test frame
        f = f[:self.npixels*1, 45:self.npixels*16+45].ravel()
        fe = fitsio.read(self.fnames[self.testframe], ext=2)[:self.npixels*1, 45:self.npixels*16+45].ravel()
        mask = bkg_aper[:self.npixels, 45:45+self.npixels*16].ravel()
        sigma_w_inv = self.Xf.T[:, mask].dot(self.Xf[mask]).toarray()
        B = self.Xf.T[:, mask].dot(f[mask])
        w = np.linalg.solve(sigma_w_inv, B)
        mod = self.Xf.dot(w)
        chi = (f - mod)**2/fe**2

        mask &= (chi < np.percentile(chi, 95))

        sigma_w_inv = self.Xf.T[:, mask].dot(self.Xf[mask]).toarray()
        XfTk = self.Xf.T[:, mask]
        self.w = np.zeros((self.nfiles, self.Xf.shape[1]))
        for idx, fname in enumerate(tqdm(self.fnames, desc='Calculating Background Correction')):
            f = fitsio.read(fname, ext=1)[:self.npixels*1, 45:self.npixels*16+45].ravel()
            fe = fitsio.read(fname, ext=2)[:self.npixels*1, 45:self.npixels*16+45].ravel()
            B = XfTk.dot(f[mask])
            self.w[idx] = np.linalg.solve(sigma_w_inv, B)


    def _get_X(self, c, r):
        """Make the X array we'll use for linear algebra """
        Xcf = lk.designmatrix.create_sparse_spline_matrix(c, knots=np.linspace(45, 2048+45, 82)[1:-1], degree=3).append_constant().X.tocsr()
        Xcf = sparse.vstack([Xcf for idx in range(128)])
        Xrf = lk.designmatrix.create_sparse_spline_matrix(r, knots=np.linspace(0, 128, 6)[1:-1], degree=3).append_constant().X.tocsr()
        Xrf = sparse.hstack([Xrf for idx in range(2048)]).reshape((Xcf.shape[0], Xrf.shape[1])).tocsr()
        Xf = sparse.hstack([Xrf.multiply(X.T) for X in Xcf.T]).tocsr()
        return Xf


    def __repr__(self):
        return f'BackgroundCorrector [{self.nfiles}]'


    def _get_star_aperture(self, fname, magnitude_limit):
        """
        fname: path to a TESS FFI to use as a test
        """

        # Find the date, we assume that people haven't renamed files...
        datekey = fname.split('/tess')[1][:7]


        def _get():
            """Helper, rips the mask, dra, ddec and g flux out of a chunk of data"""
            ms = []
            dras = []
            ddecs = []
            phots = []
            npixs = np.interp(gd.phot_g_mean_mag.value, _christina_guess[:, 0], _christina_guess[:, 1])
            for idx in np.arange(0, len(gd)):
                dra = (ra - gd.ra[idx].value)
                ddec = (dec - gd.dec[idx].value)
                r = (dra**2 + ddec**2)**0.5
                m = np.where(r < (27/3600 * npixs[idx]))
                dras.append(dra[m])
                ddecs.append(ddec[m])
                phots.append(np.ones(len(m[0])) * gd.phot_g_mean_flux[idx].value)
                ms.append(m)
            return np.hstack(ms), np.hstack(dras), np.hstack(ddecs), np.hstack(phots)


        row, column = np.mgrid[:2078, :2136]
        column, row = column[:2048, 45:2048+45], row[:2048, 45:2048+45]

        locs, dras, ddecs, phots = [], [], [], []
        for idx in range(2048//self.npixels):
            for jdx in tqdm(range(2048//self.npixels)):
                ra, dec = self.wcs.all_pix2world(column[self.npixels*idx:self.npixels*(idx+1), self.npixels*jdx:self.npixels*(jdx+1)].ravel(),
                                            row[self.npixels*idx:self.npixels*(idx+1), self.npixels*jdx:self.npixels*(jdx+1)].ravel(), 0)
                ra, dec = ra.reshape((self.npixels, self.npixels)), dec.reshape((self.npixels, self.npixels))
                gd = query_gaia_sources(ra.mean(),
                                        dec.mean(),
                                        np.hypot(ra - ra.mean(), dec - dec.mean()).max(),
                                        epoch=Time.strptime(datekey, '%Y%j').jyear, magnitude_limit=magnitude_limit)
                ms, dra, ddec, phot = _get()
                ms[0] += self.npixels*idx
                ms[1] += self.npixels*jdx

                locs.append(ms)
                dras.append(dra)
                ddecs.append(ddec)
                phots.append(phot)
            # For testing!
            break
        return np.hstack(locs), np.hstack(dras), np.hstack(ddecs), np.hstack(phots)
