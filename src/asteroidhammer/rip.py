"""Tools to rip asteroids out of TESS data"""
import numpy as np
import tess_ephem as te

import copy
from scipy import sparse
from tqdm import tqdm
import fitsio
import lightkurve as lk

from fbpca import pca

from astropy.io import fits
import pandas as pd

from astropy.stats import sigma_clip
from astropy.time import Time
import astropy.units as u
from . import PACKAGEDIR


class Ripper(object):
    """Class to rip aseroids"""

    def __init__(
        self,
        sector,
        catalog_file=None,
        ephemerides_file=None,
        asteroid_names_file=None,
        ccds=[3, 4],
        cameras=[1],
        magnitude_limit=20,
        n_asteroids=100,
    ):
        self.catalog_file = catalog_file
        self.sector = sector
        self.ccds = ccds
        self.cameras = cameras
        self.magnitude_limit = magnitude_limit
        self.time = np.loadtxt(
            f"{PACKAGEDIR}/data/sector{sector:04}_camera{cameras[0]}_ccd{ccds[0]}.txt"
        )

        if ephemerides_file is None:
            if catalog is None:
                raise ValueError("Please pass either `ephemerides` or `catalog`")
            catalog = pd.read_csv(catalog_file)
            catalog = catalog[["pdes", "full_name", "min_Vmag", "sector"]][
                catalog.sector == self.sector
            ].sort_values("min_Vmag", ascending=False)
            catalog = catalog[
                (catalog.min_Vmag <= magnitude_limit) & (catalog.min_Vmag != 0)
            ][::-1].reset_index(drop=True)
            if n_asteroids is not None:
                n_asteroids = len(catalog)
            eps = []
            for adx, d in tqdm(
                df[:n_asteroids].iterrows(),
                total=n_asteroids,
                desc="Downloading asteroid ephemerides",
            ):
                ep = te.ephem(d.pdes, Time(time, format="jd")).reset_index()
                ep.time = Time(ep["time"]).value
                ep["pdes"] = d.pdes
                eps.append(ep)
            eps = pd.concat(eps)
            u, i = np.unique(np.asarray(eps["pdes"], str), return_inverse=True)
            pd.DataFrame(u).to_csv(f"sec{sector:04}_asteroidnames.csv", index=False)
            eps["pdes_idx"] = i
            eps[["time", "camera", "ccd", "column", "row", "pdes_idx"]].to_parquet(
                f"sec{sector:04}_ephemerides.parquet"
            )
            self.ephemerides_file = f"sec{self.sector:04}_ephemerides.parquet"
            self.asteroid_names_file = f"sec{sector:04}_asteroidnames.csv"
        else:
            self.ephemerides_file = ephemerides_file
            self.asteroid_names_file = asteroid_names_file
        self.eps = pd.read_parquet(self.ephemerides_file)
        self.names = pd.read_csv(self.asteroid_names_file)

    def __repr__(self):
        return f"Ripper [sec: {self.sector}, cams: {self.cameras}, ccds: {self.ccds}]"

    def run(
        self,
        dir=".",
        ntimes=30,
        npca_components=5,
        nsamp=50,
        aperture_radius=1,
        nmask=5,
    ):

        """
        Parameters:
        ----------
        dir : str
            Directory containing the background cube files
        ntimes: int
            Number of times before and after asteroid to use to model background
        npca_components : int
            Number of pca components to use to model background
        nsamp : int
            Number of samples to generate when finding errors for light curve
        aperture_radius: int
            Radius of the aperture in pixels. If 1, the aperture will be 3x3. If 2, aperture will be 5x5.
        clip_corners: bool
            Whether to clip out the corners of the aperture to make it more circular.
        nmask: int
            Number of points before and after asteroid time to mask out during fitting. Christina will make this an automatic setting.
        """

        # for camera in self.cameras:
        #     for ccd in self.ccds:
        raise NotImplementedError()

    def _get_scattered_light(self, time, col, row, np1=1):
        """Get the scattered light at all times for a list of cols and rows

        Parameters:
        ----------
        col : np.ndarray of ints
            Array of the closest integer columns, at times object is in the dataset
        row : np.ndarray of ints
            Array of the closest integer rows, at times object is in the dataset

        Returns:
        bkg : np.ndarray of floats
            ...
        """
        l = np.where(self.time == time[0])[0][0]
        # List of tuples of all the coordinates
        coords = [
            (r + r1, c + c1)
            for c, r in zip(col, row)
            for c1 in np.arange(-np1, np1 + 1)
            for r1 in np.arange(-np1, np1 + 1)
        ]
        # Unique coordinates
        unq_coords = list(set(coords))
        # Where is each coordinate in the (2048, 2048) FFI
        locs = []
        for coord in unq_coords:
            if (coord[0] > 0) & (coord[0] < 2048) & (coord[1] > 0) & (coord[1] < 2048):
                locs.append(np.ravel_multi_index(coord, (2048, 2048)))
            else:
                locs.append(0)
        # Find the background at all times, at those unique coordinates.
        Xf = self.b.Xf[locs].toarray()
        bkg1 = Xf.dot(self.b.w[np.arange(len(col)) + l].T)

        # Some of the coordinates are duplicates, so we need to duplicate those entries
        s = np.asarray([f"{l[0]},{l[1]}" for l in coords])
        i, j, k = np.unique(s, return_counts=True, return_inverse=True)
        bkg1 = bkg1[j]

        bkg = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(col), len(col))) * np.nan
        jdx = 0
        for idx, c, r in zip(range(len(col)), col, row):
            for c1 in np.arange(-np1, np1 + 1):
                for r1 in np.arange(-np1, np1 + 1):
                    bkg[r1 + np1, c1 + np1, :, idx] = bkg1[jdx]
                    jdx += 1
        return bkg

    def _get_jitter_correction(
        self, dat, bkg, tmask, Ts, np1=1, npca_components=5, npoly=2
    ):

        t1 = (self.time - self.time.mean()) / (self.time.max() - self.time.min())
        poly = np.vstack([t1 ** idx for idx in range(npoly + 1)]).T

        res = dat - bkg
        jitter_model = np.zeros_like(dat) * np.nan
        jitter_err = np.zeros_like(dat) * np.nan
        for tdx in tqdm(range(dat.shape[2]), desc="Applying Correction"):
            t = Ts[np1, np1, :, tdx]
            k = np.isfinite(t)
            t = t[k]
            if len(t) == 0:
                continue
            X = np.hstack(
                [pca(dmed[t.astype(int), :], npca_components)[0], poly[t.astype(int)]]
            )
            j = tmask[np1, np1, k, tdx]
            if j.sum() < 15:
                continue

            sigma_w_inv = X[j].T.dot(X[j])

            werr = np.linalg.inv(sigma_w_inv)

            for c1 in np.arange(-np1, np1 + 1):
                for r1 in np.arange(-np1, np1 + 1):
                    w = np.linalg.solve(sigma_w_inv, X[j].T.dot(res[r1, c1, k, tdx][j]))
                    jitter_model[r1, c1, k, tdx] = X.dot(w)
                    w_samp = np.random.multivariate_normal(w, werr, size=(nsamp))
                    jitter_err[r1, c1, k, tdx] = np.asarray(
                        [X.dot(w_samp1) for w_samp1 in w_samp]
                    ).std(axis=0)
        return jitter_model, jitter_err


class Asteroid(object):
    """ A class to hold Christina's unusual data products"""

    def __init__(
        self,
        name,
        time,
        col,
        row,
        lag_time=0.25,
    ):
        self.name = name
        self.time = time * u.day
        self.col = col * u.pixel
        self.row = row * u.pixel
        self.ncadences = len(col)
        self.speed = np.gradient(np.hypot(self.col, self.row)) / np.gradient(self.time)
        self.lag_time = lag_time * u.day
        self._lead_sep = (
            (self.loc[0] - self.lead[0]) ** 2 + (self.loc[1] - self.lead[1]) ** 2
        ) ** 0.5
        self._lag_sep = (
            (self.loc[0] - self.lead[0]) ** 2 + (self.loc[1] - self.lead[1]) ** 2
        ) ** 0.5

    @property
    def loc(self):
        return (self.col, self.row)

    @property
    def lead(self):
        return (
            np.gradient(self.col) / np.gradient(self.time) * self.lag_time + self.col,
            np.gradient(self.row) / np.gradient(self.time) * self.lag_time + self.row,
        )

    @property
    def lag(self):
        return (
            np.gradient(self.col) / np.gradient(self.time) * -self.lag_time + self.col,
            np.gradient(self.row) / np.gradient(self.time) * -self.lag_time + self.row,
        )

    def copy(self):
        return copy.deepcopy(self)

    def __add__(self, other):
        new_ast = self.copy()
        if not hasattr(self, "dat"):
            raise ValueError("Please populate data before doing math")
        new_ast.dat = self.dat + other
        return new_ast

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return (-1 * self).__add__(other)

    def __mul__(self, other):
        new_ast = self.copy()
        if not hasattr(self, "dat"):
            raise ValueError("Please populate data before doing math")
        new_ast.dat = self.dat * other
        return new_ast

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __rtruediv__(self, other):
        new_ast = self.copy()
        if not hasattr(self, "dat"):
            raise ValueError("Please populate data before doing math")
        new_ast.dat = self.dat / other
        return new_ast

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    @property
    def tpf(self):
        raise NotImplementedError

    @property
    def lc(self):
        return lk.LightCurve(
            time=self.time,
            flux=np.nansum(
                self.dat[1, :, :, np.diag(np.ones(self.ncadences, bool))], axis=(1, 2)
            ),
            label=self.name,
            targetid=self.name,
        )

    @property
    def lag_lc(self):
        return lk.LightCurve(
            time=self.time - self.lag_time,
            flux=np.nansum(
                self.dat[0, :, :, np.diag(np.ones(self.ncadences, bool))], axis=(1, 2)
            ),
            label=self.name + " [lagged]",
            targetid=self.name,
        )

    @property
    def lead_lc(self):
        return lk.LightCurve(
            time=self.time + self.lag_time,
            flux=np.nansum(
                self.dat[2, :, :, np.diag(np.ones(self.ncadences, bool))], axis=(1, 2)
            ),
            label=self.name + " [leading]",
            targetid=self.name,
        )

    @staticmethod
    def from_name(self, name, sector=None):
        raise NotImplementedError

    def download(
        self, times, fnames, aperture_radius=1, cadence_padding=30, clip_corners=False
    ):
        l = np.where(times == self.time[0].value)[0][0]
        self.dat, self.cols, self.rows, self.Ts = self._get_asteroid_from_local_files(
            fnames,
            l,
            np1=aperture_radius,
            nb=cadence_padding,
            clip_corners=clip_corners,
        )
        self.tmask = self._get_asteroid_mask(np1=aperture_radius, nmask=5)
        return self

    def __repr__(self):
        return f"Asteroid {self.name}"

    def _get_asteroid_from_local_files(
        self, fnames, l, np1=1, nb=30, clip_corners=False
    ):
        """Get the data from local files

        Parameters:
        ----------
        fnames : list of str
            Filenames of local TESS ffi files for a given camera/ccd
        time : np.ndarray of floats
            time at each col and row
        col : np.ndarray of ints
            Array of the closest integer columns, at times object is in the dataset
        row : np.ndarray of ints
            Array of the closest integer rows, at times object is in the dataset

        """

        dat = (
            np.zeros((3, np1 * 2 + 1, np1 * 2 + 1, self.ncadences, self.ncadences))
            * np.nan
        )
        Ts = (
            np.zeros((3, np1 * 2 + 1, np1 * 2 + 1, self.ncadences, self.ncadences))
            * np.nan
        )
        cols = (
            np.zeros((3, np1 * 2 + 1, np1 * 2 + 1, self.ncadences, self.ncadences))
            * np.nan
        )
        rows = (
            np.zeros((3, np1 * 2 + 1, np1 * 2 + 1, self.ncadences, self.ncadences))
            * np.nan
        )

        for tdx in tqdm(
            np.arange(0, self.ncadences), desc="Extracting asteroid from FFIs"
        ):
            tmask = np.arange(
                np.max([tdx - nb, 0]), np.min([tdx + nb + 1, self.ncadences])
            )
            with fitsio.FITS(fnames[tdx + l]) as fts:
                for c1 in np.arange(-np1, np1 + 1):
                    for r1 in np.arange(-np1, np1 + 1):
                        if clip_corners:
                            if (np.abs(c1) == np1) & (np.abs(r1) == np1):
                                continue
                        ldx = 0
                        for col, row in [self.lag, self.loc, self.lead]:
                            c = col.value[tmask] + c1
                            r = row.value[tmask] + r1
                            for idx in range(len(c)):
                                if (
                                    ((c[idx] - 45) < 0)
                                    | ((r[idx]) < 0)
                                    | ((c[idx] - 45 + 1) >= 2048)
                                    | ((r[idx] + 1) >= 2048)
                                ):
                                    continue
                                cols[ldx, r1 + np1, c1 + np1, tdx, tmask[idx]] = c[idx]
                                rows[ldx, r1 + np1, c1 + np1, tdx, tmask[idx]] = r[idx]
                                dat[ldx, r1 + np1, c1 + np1, tdx, tmask[idx]] = fts[1][
                                    r[idx] : r[idx] + 1, c[idx] : c[idx] + 1
                                ]
                                Ts[ldx, r1 + np1, c1 + np1, tdx, tmask[idx]] = tdx + l
                            ldx += 1
        return dat, cols, rows, Ts

    def _get_asteroid_from_s3(self):
        """Get the data from s3"""
        raise NotImplementedError()

    def _get_asteroid_mask(self, np1=1, nmask=5):
        # Masks out asteroid...
        tmask = np.ones(self.dat.shape, bool)
        for c1 in np.arange(-np1, np1 + 1):
            for r1 in np.arange(-np1, np1 + 1):
                for kdx in np.arange(-nmask, nmask + 1):
                    tmask[1, r1 + np1, c1 + np1,] &= ~np.diag(
                        np.ones(self.dat.shape[3], dtype=bool), k=kdx
                    )[np.abs(kdx) :, np.abs(kdx) :]
        tmask &= np.isfinite(self.dat)
        return tmask


#
# nast = 100
# sector = 2
# df = pd.read_csv("../../../catalog/first_big_catalog.csv.zip")
#
#
# eps = []
# for adx, d in tqdm(
#     df[:nast].iterrows(), total=nast, desc="Downloading asteroid ephemerides"
# ):
#     ep = te.ephem(d.pdes, Time(time, format="jd")).reset_index()
#     ep.time = Time(ep["time"]).value
#     ep["pdes"] = d.pdes
#     eps.append(ep)
#
# eps = pd.concat(eps)
#
#
# nb = 30
# npca_components = 5
# nsamp = 50
#
# np1 = 2
# clip_corners = False
# nmask = 5
#
#
# ccd = 3
# camera = 1
#
# # b = ah.BackgroundCorrector.load(f'sec{sector:04}_camera{camera}_ccd{ccd}.tessbkg_lite')
#
# # time = np.zeros(len(b.fnames))
# # for idx, fname in enumerate(tqdm(b.fnames)):
# #     hdr = fitsio.read_header(fname)
# #     time[idx] = (hdr["TSTART"] + hdr["TSTOP"]) / 2 + 2457000
# # time = np.sort(time)
#
# # dmed = b.dmed[:, b.dmed[0] != 1]
# for pde in eps[(eps["ccd"] == ccd) & (eps["camera"] == camera)].pdes.unique():
#     ep = eps[eps.pdes == pde]
#     ep = ep[ep.ccd == ccd]
#
#     col, row = np.asarray((np.round(ep.column) - 1), int), np.asarray(
#         (np.round(ep.row) - 1), int
#     )
#     l = np.where(time == ep.time[0])[0][0]
#
#     dat = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
#     bkg = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
#     Ts = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
#     cols = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
#     rows = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
#
#     l = np.where(time == ep.time[0])[0][0]
#
#     for tdx in tqdm(np.arange(0, len(ep)), desc="Extracting asteroid from FFIs"):
#         tmask = np.arange(np.max([tdx - nb, 0]), np.min([tdx + nb + 1, len(ep)]))
#         with fitsio.FITS(b.fnames[tdx + l]) as fts:
#             for c1 in np.arange(-np1, np1 + 1):
#                 for r1 in np.arange(-np1, np1 + 1):
#                     if clip_corners:
#                         if (np.abs(c1) == np1) & (np.abs(r1) == np1):
#                             continue
#                     c = col[tmask] + c1
#                     r = row[tmask] + r1
#                     for idx in range(len(c)):
#                         if (
#                             ((c[idx] - 45) < 0)
#                             | ((r[idx]) < 0)
#                             | ((c[idx] - 45 + 1) >= 2048)
#                             | ((r[idx] + 1) >= 2048)
#                         ):
#                             continue
#                         cols[r1 + np1, c1 + np1, tdx, tmask[idx]] = c[idx]
#                         rows[r1 + np1, c1 + np1, tdx, tmask[idx]] = r[idx]
#                         dat[r1 + np1, c1 + np1, tdx, tmask[idx]] = fts[1][
#                             r[idx] : r[idx] + 1, c[idx] : c[idx] + 1
#                         ]
#                         Ts[r1 + np1, c1 + np1, tdx, tmask[idx]] = tdx + l
#     # Masks out asteroid...
#     tmask = np.ones(dat.shape, bool)
#     for c1 in np.arange(-np1, np1 + 1):
#         for r1 in np.arange(-np1, np1 + 1):
#             for kdx in np.arange(-nmask, nmask + 1):
#                 tmask[r1 + np1, c1 + np1,] &= ~np.diag(
#                     np.ones(len(ep), dtype=bool), k=kdx
#                 )[np.abs(kdx) :, np.abs(kdx) :]
#     tmask &= np.isfinite(dat)
#
#     bkg = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
#     coords = [
#         (r + r1, c + c1)
#         for c, r in zip(col, row)
#         for c1 in np.arange(-np1, np1 + 1)
#         for r1 in np.arange(-np1, np1 + 1)
#     ]
#     unq_coords = list(set(coords))
#     # Xf = b.Xf[[np.ravel_multi_index(coord, (2048, 2048)) for coord in unq_coords]].toarray()
#
#     locs = []
#     for coord in unq_coords:
#         if (coord[0] > 0) & (coord[0] < 2048) & (coord[1] > 0) & (coord[1] < 2048):
#             locs.append(np.ravel_multi_index(coord, (2048, 2048)))
#         else:
#             locs.append(0)
#     Xf = b.Xf[locs].toarray()
#     bkg1 = Xf.dot(b.w[np.arange(len(ep)) + l].T)
#
#     s = np.asarray([f"{l[0]},{l[1]}" for l in coords])
#     i, j, k = np.unique(s, return_counts=True, return_inverse=True)
#     bkg1 = bkg1[j]
#
#     jdx = 0
#     for idx, c, r in zip(range(len(col)), col, row):
#         for c1 in np.arange(-np1, np1 + 1):
#             for r1 in np.arange(-np1, np1 + 1):
#                 bkg[r1 + np1, c1 + np1, :, idx] = bkg1[jdx]
#                 jdx += 1
#     bkg *= np.nan ** (~np.isfinite(dat))
#
#     res = dat - bkg
#     jitter_model = np.zeros_like(dat) * np.nan
#     jitter_err = np.zeros_like(dat) * np.nan
#     for tdx in tqdm(range(len(ep)), desc="Applying Correction"):
#         t = Ts[np1, np1, :, tdx]
#         k = np.isfinite(t)
#         t = t[k]
#         if len(t) == 0:
#             continue
#         X = np.hstack(
#             [pca(dmed[t.astype(int), :], npca_components)[0], poly[t.astype(int)]]
#         )
#         j = tmask[np1, np1, k, tdx]
#
#         sigma_w_inv = X[j].T.dot(X[j])
#
#         werr = np.linalg.inv(sigma_w_inv)
#
#         for c1 in np.arange(-np1, np1 + 1):
#             for r1 in np.arange(-np1, np1 + 1):
#                 w = np.linalg.solve(sigma_w_inv, X[j].T.dot(res[r1, c1, k, tdx][j]))
#                 jitter_model[r1, c1, k, tdx] = X.dot(w)
#                 w_samp = np.random.multivariate_normal(w, werr, size=(nsamp))
#                 jitter_err[r1, c1, k, tdx] = np.asarray(
#                     [X.dot(w_samp1) for w_samp1 in w_samp]
#                 ).std(axis=0)
#
#     # This time array might be obo...?
#     k = np.isfinite(Ts[np1, np1, np.diag(np.ones(len(ep), dtype=bool))])
#     lc = lk.LightCurve(
#         time[Ts[np1, np1, np.diag(np.ones(len(ep), dtype=bool))][k].astype(int)],
#         np.nansum(
#             (dat - bkg - jitter_model)[:, :, np.diag(np.ones(len(ep), dtype=bool))][
#                 :, :, k
#             ],
#             axis=(0, 1),
#         ),
#         np.nansum(
#             (jitter_err ** 2)[:, :, np.diag(np.ones(len(ep), dtype=bool))][:, :, k],
#             axis=(0, 1),
#         )
#         ** 0.5,
#         targetid=pde,
#         label=pde,
#     )
#     lc.to_fits(
#         f"/Users/ch/Projects/PDART/asteroid_lightcurves/fits/{pde}.fits",
#         overwrite="True",
#     )
