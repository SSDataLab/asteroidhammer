"""tools to remove the background from TESS FFIs in a fast, light weight way"""
import fitsio
import numpy as np
from tqdm import tqdm
from scipy import sparse
import pandas as pd

import os
from glob import glob
import pickle

import lightkurve as lk

from fbpca import pca

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats

import matplotlib.pyplot as plt

from .query import query_gaia_sources

import pyarrow as pa
import pyarrow.parquet as pq

import tess_ephem
from scipy.interpolate import interp1d


def _count_unq(locs):
    """Finds unique tuples"""
    s = np.asarray([f"{l[0]},{l[1]}" for l in locs.T])
    i, j, k = np.unique(s, return_counts=True, return_inverse=True)
    return k[j]


_keys = [
    "fnames",
    "nfiles",
    "nchunks",
    "npixels",
    "magnitude_limit",
    "testframe",
    "camera",
    "ccd",
    "sector",
    "wcs",
    "star_aper",
    "mask",
    "bright_mask",
    "bkg_aper",
    "avg",
    "w",
    "dra_bin",
    "ddec_bin",
    "med",
    "dmed",
]
_keys_lite = [
    "fnames",
    "nfiles",
    "nchunks",
    "npixels",
    "magnitude_limit",
    "testframe",
    "camera",
    "ccd",
    "sector",
    "w",
    "dra_bin",
    "ddec_bin",
    "med",
    "dmed",
]


## Array for circular apertures as a function of g magnitude...
_christina_guess = np.asarray(
    [
        [0, 15],
        [1, 15],
        [2, 12],
        [3, 10],
        [4, 8],
        [5, 6],
        [6, 5],
        [8, 4],
        [10, 4],
        [11, 3],
        [12, 3],
        [13, 3],
        [14, 2],
        [15, 2],
        [16, 1],
        [17, 1],
        [18, 1],
    ]
)


class BackgroundCorrector(object):
    def __init__(
        self,
        fnames=None,
        nchunks=16,
        testframe=None,
        magnitude_limit=17,
        gaia_file=None,
        asteroid_catalog_file=None,
    ):
        """Parameters

        fnames : list of str
            List containing paths to the FFI files
        nchunks : int
            Number of chunks to split the data into when querying Gaia
        """
        self.fnames = fnames
        self.gaia_file = gaia_file
        self.asteroid_catalog_file = asteroid_catalog_file

        if not os.path.isdir(".ah_cache"):
            os.mkdir(".ah_cache")
        # This is a work around so I can "load"...it's not ideal.
        if fnames is not None:
            self.nfiles = len(fnames)
            self.nchunks = int(nchunks)
            self.npixels = 2048 // self.nchunks
            self.magnitude_limit = magnitude_limit

            if testframe is None:
                testframe = len(fnames) // 2
            self.testframe = testframe
            h = fitsio.read_header(self.fnames[self.testframe], 1)
            self.ccd = h["CCD"]
            self.camera = h["CAMERA"]
            # assuming no one has renamed files....
            self.sector = int(self.fnames[self.testframe].split("-s")[1].split("-")[0])

            self.wcs = WCS(fits.open(self.fnames[self.testframe])[1])

            row, column = np.mgrid[:2078, :2136]
            column, row = column[:2048, 45 : 2048 + 45], row[:2048, 45 : 2048 + 45]
            ra, dec = self.wcs.all_pix2world(column.ravel(), row.ravel(), 0)
            ra, dec = ra.reshape(column.shape), dec.reshape(column.shape)
            self._get_star_aperture(
                self.fnames[self.testframe], magnitude_limit=self.magnitude_limit
            )

            count = _count_unq(self.locs)

            self.star_aper = np.zeros((2078, 2136), bool)
            self.star_aper[self.locs[0], self.locs[1] + 45] = True
            self.mask = (
                (self.mag < 13)
                & (count == 1)
                & (
                    np.hypot(self.dra, self.ddec)
                    <= 2.0 * np.median(np.hypot(np.diff(ra[0]), np.diff(dec[:, 0])))
                )
            )
            self.bright_mask = np.zeros((2078, 2136), bool)
            self.bright_mask[
                self.locs[0][self.mask], self.locs[1][self.mask] + 45
            ] = True
            self.bright_mask = self.bright_mask[:2048, 45 : 2048 + 45].ravel()

            self.bkg_aper = ~self.star_aper
            self.bkg_aper[2048:] = False
            self.bkg_aper[:, :45] = False
            self.bkg_aper[:, 2048 + 45 :] = False

            if not os.path.isfile(
                f".ah_cache/sec{self.sector:04}_camera{self.camera}_ccd{self.ccd}_bright_mask.npz"
            ):
                time = np.zeros(len(fnames))
                for idx, fname in enumerate(tqdm(fnames)):
                    hdr = fitsio.read_header(fname)
                    time[idx] = (hdr["TSTART"] + hdr["TSTOP"]) / 2 + 2457000
                time = np.sort(time)

                _build_asteroid_mask(
                    time,
                    sector=self.sector,
                    camera=self.camera,
                    ccd=self.ccd,
                    catalog_file=self.asteroid_catalog_file,
                )

            self.bkg_aper &= ~self.asteroid_mask

            #            bkg_aper[self.locs[0].max() :] = False
            #            bkg_aper[:, self.locs[1].max() :] = False
            self.Xf = self._get_X(c=np.arange(2048), r=np.arange(2048))

    @property
    def asteroid_mask(self):
        asteroid_mask = np.zeros_like(self.bkg_aper)
        asteroid_mask[:2048, 45 : 2048 + 45] = sparse.load_npz(
            f".ah_cache/sec{self.sector:04}_camera{self.camera}_ccd{self.ccd}_bright_mask.npz"
        ).toarray()
        return asteroid_mask

    @property
    def _cache_fnames(self):
        if self.gaia_file == None:
            if os.path.isdir(".ah_cache"):
                fnames = [
                    (
                        ".ah_cache/"
                        + self.fnames[0].split("/")[-1].split(".fits")[0]
                        + f"_{idx}_{jdx}_{self.magnitude_limit}mag.parquet"
                    )
                    for idx in range(self.nchunks)
                    for jdx in range(self.nchunks)
                ]
            else:
                raise ValueError("No gaia information? Provide a gaia file.")
            return [fname for fname in fnames if os.path.isfile(fname)]
        return [self.gaia_file]

    @property
    def locs(self):
        if len(self._cache_fnames) == 0:
            return None
        return np.vstack(
            [
                pd.read_parquet(
                    file,
                    columns=["locs0", "locs1"],
                )
                for file in self._cache_fnames
                if os.path.isfile(file)
            ]
        ).T.astype(int)

    @property
    def dra(self):
        if len(self._cache_fnames) == 0:
            return None
        return np.vstack(
            [
                pd.read_parquet(
                    file,
                    columns=["dra"],
                )
                for file in self._cache_fnames
                if os.path.isfile(file)
            ]
        )[:, 0]

    @property
    def ddec(self):
        if len(self._cache_fnames) == 0:
            return None
        return np.vstack(
            [
                pd.read_parquet(
                    file,
                    columns=["ddec"],
                )
                for file in self._cache_fnames
                if os.path.isfile(file)
            ]
        )[:, 0]

    @property
    def phot(self):
        if len(self._cache_fnames) == 0:
            return None
        return np.vstack(
            [
                pd.read_parquet(
                    file,
                    columns=["phot"],
                )
                for file in self._cache_fnames
                if os.path.isfile(file)
            ]
        )[:, 0]

    @property
    def mag(self):
        # # Christina's empirical calibration
        gaia_zp = 25.688365751
        return gaia_zp + -2.5 * np.log10(self.phot)

    def find_bkg_solution(self, nbins=(20, 20)):
        # Clip out outliers present in the test frame
        f = fitsio.read(self.fnames[self.testframe], ext=1)
        # Saturation
        sat_mask = f[:2048, 45 : 2048 + 45] > 9e4
        for count in range(4):
            sat_mask |= np.gradient(sat_mask.astype(float))[0] != 0

        f = f[:2048, 45 : 2048 + 45].ravel()
        fe = fitsio.read(self.fnames[self.testframe], ext=2)[
            :2048, 45 : 2048 + 45
        ].ravel()
        mask = self.bkg_aper.copy()[:2048, 45 : 2048 + 45].ravel()
        mask &= ~sat_mask.ravel()
        mask &= f < np.nanpercentile(f[mask], 99.99)
        mask &= f > np.nanpercentile(f[mask], 0.001)

        # Prior mu is all zero...
        prior_sigma = 1 / np.zeros(self.Xf.shape[1])
        prior_sigma[-2048:] = 500
        #        prior_sigma = np.ones(self.Xf.shape[1]) * 3000

        for sigma in [10, 5]:
            sigma_w_inv = self.Xf.T[:, mask].dot(self.Xf[mask]).toarray() + np.diag(
                1 / prior_sigma ** 2
            )
            B = self.Xf.T[:, mask].dot(f[mask])
            w = np.linalg.solve(sigma_w_inv, B)
            mod = self.Xf.dot(w)
            res = f - mod
            # Clip out heinous outliers
            # chi = (f - mod) ** 2 / fe ** 2
            #            mask &= chi < np.percentile(chi, perc)
            stats = sigma_clipped_stats(np.ma.masked_array(res, ~mask), sigma=sigma)
            mask &= (res - stats[1]) / (stats[2]) < 4

        sigma_w_inv = self.Xf.T[:, mask].dot(self.Xf[mask]).toarray() + np.diag(
            1 / prior_sigma ** 2
        )
        XfTk = self.Xf.T[:, mask]
        B = XfTk.dot(f.ravel()[mask])

        self.avg = f - self.Xf.dot(np.linalg.solve(sigma_w_inv, B))

        self.avg_br = self.avg[self.bright_mask]
        self.avg = self.avg.reshape((2048, 2048))

        self.w = np.zeros((self.nfiles, self.Xf.shape[1]))
        Xf_bright = self.Xf[self.bright_mask]

        self.dra_bin, self.ddec_bin, bin_masks = self._get_bin_basis(nbins=nbins)
        self.med = np.asarray(
            [np.nanmedian(self.avg_br[m1]) for m1 in bin_masks]
        ).reshape(nbins)

        #        self.avg = np.zeros(self.bright_mask.sum())
        self.dmed = np.ones((self.nfiles, *nbins))
        for idx, fname in enumerate(
            tqdm(self.fnames, desc="Calculating Background Correction")
        ):
            f = fitsio.read(fname, ext=1)[:2048, 45 : 2048 + 45] - self.avg
            # fe = fitsio.read(fname, ext=2)[:self.npixels*1, 45:self.npixels*16+45]

            B = XfTk.dot(f.ravel()[mask])
            self.w[idx] = np.linalg.solve(sigma_w_inv, B)

            # Need to extract stars here
            f_bright = f.ravel()[self.bright_mask] - Xf_bright.dot(self.w[idx])

            self.dmed[idx] = np.asarray(
                [
                    np.nanmedian((f_bright[m1] + self.avg_br[m1]) / self.avg_br[m1])
                    if m1.sum() != 0
                    else 1
                    for m1 in bin_masks
                ]
            ).reshape(nbins)
            # table = pa.table({'flux': f_bright})
            # pq.write_table(table, f'.database/{idx:04}.parquet')
        #            self.avg += f_bright
        #        self.avg /= self.nfiles
        return self

    def _get_bin_basis(self, nbins=(20, 20)):
        s = np.lexsort((self.locs[1][self.mask], self.locs[0][self.mask]))
        dra, ddec = self.dra[self.mask][s], self.ddec[self.mask][s]
        nbins1, nbins2 = nbins
        x, y = np.linspace(dra.min(), dra.max(), nbins1), np.linspace(
            ddec.min(), ddec.max(), nbins2
        )
        x = np.hstack([x, x[-1] + np.median(np.diff(x))])
        y = np.hstack([y, y[-1] + np.median(np.diff(y))])
        # x += np.median(np.diff(x))/2
        # y += np.median(np.diff(y))/2
        m = []
        for idx, x1 in enumerate(x[:-1]):
            for jdx, y1 in enumerate(y[:-1]):
                m.append(
                    (dra > x1) & (dra < x[idx + 1]) & (ddec > y1) & (ddec < y[jdx + 1])
                )
        return x[:-1] - np.median(np.diff(x)) / 2, y[:-1] - np.median(np.diff(y)) / 2, m

    def save(self, lite=True):
        if lite:
            out = f"sec{self.sector:04}_camera{self.camera}_ccd{self.ccd}.tessbkg_lite"
            d = {key: getattr(self, key) for key in _keys_lite if hasattr(self, key)}
        else:
            out = f"sec{self.sector:04}_camera{self.camera}_ccd{self.ccd}.tessbkg"
            d = {key: getattr(self, key) for key in _keys if hasattr(self, key)}
        pickle.dump(d, open(out, "wb"))
        # this should be parquet
        out = f"sec{self.sector:04}_camera{self.camera}_ccd{self.ccd}.gaia"
        pd.concat([pd.read_parquet(file) for file in self._cache_fnames]).to_parquet(
            out
        )
        return self

    @staticmethod
    def load(bkg_filename, gaia_file=None):
        """Load a saved fit"""
        b = BackgroundCorrector(gaia_file=gaia_file)
        # this should be parquet
        d = pickle.load(open(bkg_filename, "rb"))
        _ = [setattr(b, key, d[key]) for key in _keys if key in d.keys()]
        b.Xf = b._get_X(c=np.arange(2048), r=np.arange(2048))
        return b

    def get_full_bkg_frame(self, frames=0):
        if not hasattr(frames, "__iter__"):
            frames = [np.copy(frames)]
        model = np.asarray(
            [self.Xf.dot(self.w[frame]).reshape((2048, 2048)) for frame in frames]
        )
        if len(frames) == 1:
            return model[0]
        return model

    def get_bkg_cutout(self, loc=[[45, 128], [52, 90]], frames=0):
        # Shortened for test!
        c = np.arange(2048, dtype=int)
        r = np.arange(2048, dtype=int)

        ck = sparse.csr_matrix(((c >= loc[1][0]) & (c < loc[1][1])).astype(float)).T
        rk = sparse.csr_matrix(((r >= loc[0][0]) & (r < loc[0][1])).astype(float)).T
        ck = sparse.vstack([ck for idx in range(2048)], format="csr")
        rk = (
            sparse.hstack([rk for idx in range(2048)])
            .reshape((ck.shape[0], rk.shape[1]))
            .tocsr()
        )
        Xf = self.Xf.multiply(ck).multiply(rk)
        # Only bother with pixels that actually have values
        kw = np.asarray(Xf.sum(axis=1) != 0)[:, 0]
        if not hasattr(frames, "__iter__"):
            frames = [np.copy(frames)]
        model = np.asarray(
            [
                Xf[kw].dot(self.w[frame]).reshape(tuple(np.diff(loc)[:, 0]))
                for frame in frames
            ]
        )
        if len(frames) == 1:
            return model[0]
        return model

    def _get_X(self, c, r):
        """Make the X array we'll use for linear algebra """
        Xcf = (
            lk.designmatrix.create_spline_matrix(
                c, knots=list(np.linspace(0, 2048, 82)[1:-1]), degree=3
            )
            .append_constant()
            .to_sparse()
            .X.tocsr()
        )

        Xcf = sparse.vstack([Xcf for idx in range(2048)])
        Xrf = (
            lk.designmatrix.create_spline_matrix(
                r, knots=list(np.linspace(0, 2048, 82)[1:-1]), degree=3
            )
            .append_constant()
            .to_sparse()
            .X.tocsr()
        )
        Xrf = (
            sparse.hstack([Xrf for idx in range(2048)])
            .reshape((Xcf.shape[0], Xrf.shape[1]))
            .tocsr()
        )

        Xf = sparse.hstack([Xrf.multiply(X.T) for X in Xcf.T]).tocsr()

        # Straps
        diag = np.diag(np.ones(2048))
        e = sparse.lil_matrix((2048, 2048 * 2048))
        for idx in range(2048):
            e[idx, np.arange(2048) * 2048 + idx] = 1
        X = e.T.tocsr()

        Xf = sparse.hstack([Xf, X]).tocsr()
        return Xf

    def __repr__(self):
        return f"BackgroundCorrector [Sector {self.sector}, Camera {self.camera}, CCD {self.ccd}]"

    def _get_star_aperture(self, fname, magnitude_limit):
        """
        fname: path to a TESS FFI to use as a test
        """

        # This might not be advisable?
        if not os.path.isdir(".ah_cache"):
            os.mkdir(".ah_cache")

        # Find the date, we assume that people haven't renamed files...
        datekey = fname.split("/tess")[1][:7]

        def _get():
            """Helper, rips the mask, dra, ddec and g flux out of a chunk of data"""
            ms = []
            dras = []
            ddecs = []
            phots = []
            npixs = np.interp(
                gd.phot_g_mean_mag.value, _christina_guess[:, 0], _christina_guess[:, 1]
            )
            for idx in np.arange(0, len(gd)):
                dra = ra - gd.ra[idx].value
                ddec = dec - gd.dec[idx].value
                r = (dra ** 2 + ddec ** 2) ** 0.5
                m = np.where(r < (27 / 3600 * npixs[idx]))
                dras.append(dra[m])
                ddecs.append(ddec[m])
                phots.append(np.ones(len(m[0])) * gd.phot_g_mean_flux[idx].value)
                ms.append(m)
            return np.hstack(ms), np.hstack(dras), np.hstack(ddecs), np.hstack(phots)

        row, column = np.mgrid[:2078, :2136]
        column, row = column[:2048, 45 : 2048 + 45], row[:2048, 45 : 2048 + 45]

        # locs, dras, ddecs, phots = [], [], [], []
        for idx in range(2048 // self.npixels):
            for jdx in tqdm(range(2048 // self.npixels)):
                fname = (
                    ".ah_cache/"
                    + self.fnames[0].split("/")[-1].split(".fits")[0]
                    + f"_{idx}_{jdx}_{magnitude_limit}mag.parquet"
                )
                if os.path.isfile(fname):
                    continue

                ra, dec = self.wcs.all_pix2world(
                    column[
                        self.npixels * idx : self.npixels * (idx + 1),
                        self.npixels * jdx : self.npixels * (jdx + 1),
                    ].ravel(),
                    row[
                        self.npixels * idx : self.npixels * (idx + 1),
                        self.npixels * jdx : self.npixels * (jdx + 1),
                    ].ravel(),
                    0,
                )
                ra, dec = ra.reshape((self.npixels, self.npixels)), dec.reshape(
                    (self.npixels, self.npixels)
                )

                rad = SkyCoord(
                    ra[self.npixels // 2, self.npixels // 2],
                    dec[self.npixels // 2, self.npixels // 2],
                    unit="deg",
                ).separation(SkyCoord(ra, dec, unit="deg"))

                rad = rad.deg.max()
                rad += 0.015

                gd = query_gaia_sources(
                    ra[self.npixels // 2, self.npixels // 2],
                    dec[self.npixels // 2, self.npixels // 2],
                    rad,
                    epoch=Time.strptime(datekey, "%Y%j").jyear,
                    magnitude_limit=magnitude_limit,
                )
                ms, dra, ddec, phot = _get()
                ms[0] += self.npixels * idx
                ms[1] += self.npixels * jdx

                pd.DataFrame(
                    np.vstack([ms, dra, ddec, phot]).T,
                    columns=["locs0", "locs1", "dra", "ddec", "phot"],
                ).to_parquet(fname, index=False)
            # locs.append(ms)
            # dras.append(dra)
            # ddecs.append(ddec)
            # phots.append(phot)
        # return np.hstack(locs), np.hstack(dras), np.hstack(ddecs), np.hstack(phots)
        return


def _build_asteroid_mask(
    time,
    sector,
    camera,
    ccd,
    catalog_file,
    magnitude_limit=16,
    bright_asteroid_limit=16,
    aper_size=5,
):

    """
    time in jd, nparray
    """

    if catalog_file is None:
        raise ValueError(
            "Please pass an asteroid `catalog_file` to calculate asteroid mask."
        )
    # Load catalog
    df = pd.read_csv(catalog_file)
    df = df[["pdes", "full_name", "max_Vmag"]][df.sector == sector].sort_values(
        "max_Vmag"
    )
    df = df[(df.max_Vmag <= magnitude_limit) & (df.max_Vmag != 0)][::-1].reset_index(
        drop=True
    )
    tp = time[::10]
    if not ((len(time) - 1) / 10) == int((len(time) - 1) / 10):
        tp = np.hstack([tp, time[-1]])
    tp = Time(tp, format="jd")

    bright_cap = np.where(df.max_Vmag <= bright_asteroid_limit)[0][0]

    asteroid_mask = sparse.lil_matrix((len(time), 2048 * 2048), dtype=int)
    for kdx, d in tqdm(df.iterrows(), total=len(df), desc="Building asteroid mask"):
        ep = tess_ephem.ephem(d.pdes, time=tp).reset_index()
        ep = ep[ep.camera == camera]
        ep = ep[ep.ccd == ccd]
        if len(ep) == 0:
            continue
        if len(ep) <= 5:
            ep = tess_ephem.ephem(d.pdes, time=Time(time, format="jd")).reset_index()
            ep = ep[ep.camera == camera]
            ep = ep[ep.ccd == ccd]
            col, row = ep.column, ep.row
        else:
            ep.time = Time(ep.time).value
            k = (time >= ep.time[0]) & (time <= ep.time[len(ep) - 1])
            col = interp1d(
                ep.time,
                ep.column,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )(time)
            row = interp1d(
                ep.time,
                ep.row,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )(time)
            k &= (
                (np.round(col) < 2048 + 45)
                & (np.round(col) >= 0 + 45)
                & (np.round(row) < 2048)
                & (np.round(row) >= 0)
            )

            # Mask out square around asteroid
            for idx in np.arange(-aper_size // 2, aper_size // 2):
                for jdx in np.arange(-aper_size // 2, aper_size // 2):
                    if np.abs(idx) == np.abs(jdx):
                        continue
                    c = np.max(
                        [np.round(col[k]).astype(int) + idx, np.zeros(k.sum()) + 45],
                        axis=0,
                    ).astype(int)
                    c = np.min([c, np.zeros(k.sum()) + 2047 + 45], axis=0).astype(int)

                    r = np.max(
                        [np.round(row[k]).astype(int) + jdx, np.zeros(k.sum())], axis=0
                    ).astype(int)
                    r = np.min([r, np.zeros(k.sum()) + 2047], axis=0).astype(int)
                    l = np.ravel_multi_index((c - 45, r), dims=(2048, 2048))
                    asteroid_mask[np.where(k)[0], l] = kdx
    #    plt.plot(col, row)
    asteroid_mask = asteroid_mask.tocsr()
    if not os.path.isdir(".ah_cache"):
        os.mkdir(".ah_cache")
    sparse.save_npz(
        f".ah_cache/sec{sector:04}_camera{camera}_ccd{ccd}.npz", asteroid_mask
    )

    bright_mask = sparse.csr_matrix(
        ~np.asarray(
            ((asteroid_mask > bright_cap).sum(axis=0) == 0).reshape((2048, 2048)).T
        ),
        dtype=bool,
    )
    sparse.save_npz(
        f".ah_cache/sec{sector:04}_camera{camera}_ccd{ccd}_bright_mask.npz", bright_mask
    )
    df[["pdes"]].to_csv(
        f".ah_cache/sec{sector:04}_camera{camera}_ccd{ccd}_asteroid_list.csv",
        index=False,
    )

    return asteroid_mask
