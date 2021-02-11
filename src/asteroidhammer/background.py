'''tools to remove the background from TESS FFIs in a fast, light weight way'''
import fitsio
import numpy as np
from tqdm import tqdm
from scipy import sparse

import os
from glob import glob
import pickle

import lightkurve as lk

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time

import matplotlib.pyplot as plt

from .query import query_gaia_sources

import pyarrow as pa
import pyarrow.parquet as pq


def _count_unq(locs):
    """Finds unique tuples"""
    s = np.asarray([f'{l[0]},{l[1]}' for l in locs.T])
    i, j, k = np.unique(s, return_counts=True, return_inverse=True)
    return k[j]


_keys = ['nfiles', 'nchunks', 'npixels', 'magnitude_limit', 'testframe', 'camera', 'ccd', 'sector',
        'wcs', 'locs', 'dra', 'ddec', 'phot', 'mag', 'star_aper',
        'mask', 'bright_mask', 'bkg_aper', 'avg', 'w', 'dra_bin', 'ddec_bin', 'med', 'dmed']

## Array for circular apertures as a function of g magnitude...
_christina_guess = np.asarray([[0, 10], [1, 10], [2, 10], [3, 9], [4, 7], [5, 5], [6, 4], [8, 4], [10, 4], [11, 3], [12, 3], [13, 3], [14, 2], [15, 2], [16, 2], [17, 1]])

class BackgroundCorrector(object):


    def __init__(self, fnames=None, nchunks=16, testframe=None, magnitude_limit=16, nbins=(30, 30)):
        """Parameters

        fnames : list of str
            List containing paths to the FFI files
        nchunks : int
            Number of chunks to split the data into when querying Gaia
        """
        self.fnames = fnames
        # This is a work around so I can "load"...it's not ideal.
        if fnames is not None:
            self.nfiles = len(fnames)
            self.nchunks = int(nchunks)
            self.npixels = 2048//self.nchunks
            self.magnitude_limit = magnitude_limit

            if testframe is None:
                testframe = len(fnames)//2
            self.testframe = testframe
            h = fitsio.read_header(self.fnames[self.testframe], 1)
            self.ccd = h['CCD']
            self.camera = h['CAMERA']
            # assuming no one has renamed files....
            self.sector = int(self.fnames[self.testframe].split('-s')[1].split('-')[0])

            self.wcs = WCS(fits.open(self.fnames[self.testframe])[1])

            row, column = np.mgrid[:2078, :2136]
            column, row = column[:2048, 45:2048+45], row[:2048, 45:2048+45]
            ra, dec = self.wcs.all_pix2world(column.ravel(), row.ravel(), 0)
            ra, dec = ra.reshape(column.shape), dec.reshape(column.shape)
            self.locs, self.dra, self.ddec, self.phot = self._get_star_aperture(self.fnames[self.testframe], magnitude_limit=self.magnitude_limit)



            # Christina's empirical calibration
            gaia_zp = 25.688365751
            self.mag = gaia_zp + -2.5*np.log10(self.phot)
            count = _count_unq(self.locs)

            self.star_aper = np.zeros((2078, 2136), bool)
            self.star_aper[self.locs[0], self.locs[1] + 45] = True
            self.mask = (self.mag < 13) & (count == 1) & (np.hypot(self.dra, self.ddec) <= 2.5 * np.median(np.hypot(np.diff(ra[0]), np.diff(dec[:, 0]))))
            self.bright_mask = np.zeros((2078, 2136), bool)
            self.bright_mask[self.locs[0][self.mask], self.locs[1][self.mask] + 45] = True
            # small for test
            self.bright_mask = self.bright_mask[:self.npixels*1, 45:self.npixels*16+45].ravel()


            f = fitsio.read(fnames[self.testframe], ext=1)
            bkg_aper = ~self.star_aper
            bkg_aper[f < 10] = False

            # For testing
            bkg_aper[self.npixels:] = False

            bkg_aper &= f < np.nanpercentile(f[bkg_aper], 99.5)
            bkg_aper &= f > np.nanpercentile(f[bkg_aper], 0.05)
            self.bkg_aper = bkg_aper

            # Row short for test!
            self.Xf = self._get_X(c=np.arange(2048), r=np.arange(128))

            # if os.path.isdir('.database'):
            #     [os.remove(f) for f in glob('.database/*')]
            #     os.removedirs('.database')
            # os.mkdir('.database')

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
            B = XfTk.dot(f.ravel()[mask])
            self.avg = (f - self.Xf.dot(np.linalg.solve(sigma_w_inv, B)))[self.bright_mask]


            self.w = np.zeros((self.nfiles, self.Xf.shape[1]))
            # Small for test
            Xf_bright = self.Xf[self.bright_mask]

            self.dra_bin, self.ddec_bin, bin_masks = self._get_bin_basis(nbins=nbins)
            self.med = np.asarray([np.nanmedian(self.avg[m1]) for m1 in bin_masks]).reshape(nbins)


    #        self.avg = np.zeros(self.bright_mask.sum())
            self.dmed = np.ones((self.nfiles, *nbins))
            for idx, fname in enumerate(tqdm(self.fnames, desc='Calculating Background Correction')):
                f = fitsio.read(fname, ext=1)[:self.npixels*1, 45:self.npixels*16+45]
                #fe = fitsio.read(fname, ext=2)[:self.npixels*1, 45:self.npixels*16+45]

                B = XfTk.dot(f.ravel()[mask])
                self.w[idx] = np.linalg.solve(sigma_w_inv, B)

                # Need to extract stars here
                f_bright = f.ravel()[self.bright_mask] - Xf_bright.dot(self.w[idx])
                self.dmed[idx] = np.asarray([np.nanmedian(f_bright[m1]/self.avg[m1]) if m1.sum() != 0 else 1 for m1 in bin_masks]).reshape(nbins)

                #table = pa.table({'flux': f_bright})
                #pq.write_table(table, f'.database/{idx:04}.parquet')
    #            self.avg += f_bright
    #        self.avg /= self.nfiles

    def _get_bin_basis(self, nbins=(20, 20)):
        s = np.lexsort((self.locs[1][self.mask], self.locs[0][self.mask]))
        dra, ddec = self.dra[self.mask][s], self.ddec[self.mask][s]
        nbins1, nbins2 = nbins
        x, y = np.linspace(dra.min(), dra.max(), nbins1), np.linspace(ddec.min(), ddec.max(), nbins2)
        x = np.hstack([x, x[-1] + np.median(np.diff(x))])
        y = np.hstack([y, y[-1] + np.median(np.diff(y))])
        #x += np.median(np.diff(x))/2
        #y += np.median(np.diff(y))/2
        m = []
        for idx, x1 in enumerate(x[:-1]):
            for jdx, y1 in enumerate(y[:-1]):
                m.append((dra > x1) & (dra < x[idx + 1]) & (ddec > y1) & (ddec < y[jdx + 1]))
        return x[:-1] - np.median(np.diff(x))/2, y[:-1] - np.median(np.diff(y))/2, m

    def save(self, out=None):
        if out is None:
            out = f'sec{self.sector:04}_camera{self.camera}_ccd{self.ccd}.tessbkg'
        d = {key:getattr(self, key) for key in _keys}
        # this should be parquet
        pickle.dump(d, open(out, 'wb'))

    @staticmethod
    def load(filename):
        """Load a saved fit"""
        b = BackgroundCorrector()
        # this should be parquet
        d = pickle.load(open(filename, 'rb'))
        _ = [setattr(b, key, d[key]) for key in _keys]
        b.Xf = b._get_X(c=np.arange(2048), r=np.arange(128))
        return b

    def get_full_bkg_frame(self, frames=0):
        # Smaller for test!!!
        if not hasattr(frames, '__iter__'):
            frames = [np.copy(frames)]
        model =  np.asarray([self.Xf.dot(self.w[frame]).reshape((128, 2048)) for frame in frames])
        if len(frames) == 1:
            return model[0]
        return model

    def get_bkg_cutout(self, loc=[[45, 128], [52, 90]], frames=0):
        # Shortened for test!
        c = np.arange(2048, dtype=int)
        r = np.arange(128, dtype=int)

        ck = sparse.csr_matrix(((c > loc[1][0]) & (c <= loc[1][1])).astype(float)).T
        rk = sparse.csr_matrix(((r > loc[0][0]) & (r <= loc[0][1])).astype(float)).T
        ck = (sparse.vstack([ck for idx in range(128)], format='csr'))
        rk = sparse.hstack([rk for idx in range(2048)]).reshape((ck.shape[0], rk.shape[1])).tocsr()
        Xf = self.Xf.multiply(ck).multiply(rk)
        # Only bother with pixels that actually have values
        kw = np.asarray(Xf.sum(axis=1) != 0)[:, 0]
        if not hasattr(frames, '__iter__'):
            frames = [np.copy(frames)]
        model = np.asarray([Xf[kw].dot(self.w[frame]).reshape(tuple(np.diff(loc)[:, 0])) for frame in frames])
        if len(frames) == 1:
            return model[0]
        return model

    def _get_X(self, c, r):
        """Make the X array we'll use for linear algebra """
        Xcf = lk.designmatrix.create_spline_matrix(c, knots=list(np.linspace(0, 2048, 82)[1:-1]), degree=3).append_constant().to_sparse().X.tocsr()
        Xcf = sparse.vstack([Xcf for idx in range(128)])
        Xrf = lk.designmatrix.create_spline_matrix(r, knots=list(np.linspace(0, 128, 6)[1:-1]), degree=3).append_constant().to_sparse().X.tocsr()
        Xrf = sparse.hstack([Xrf for idx in range(2048)]).reshape((Xcf.shape[0], Xrf.shape[1])).tocsr()
        Xf = sparse.hstack([Xrf.multiply(X.T) for X in Xcf.T]).tocsr()
        return Xf

    def __repr__(self):
        return f'BackgroundCorrector [Sector {self.sector}, Camera {self.camera}, CCD {self.ccd}]'

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
