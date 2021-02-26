"""Tools to rip asteroids out of TESS data"""
import numpy as np
import tess_ephem as te

from scipy import sparse
from tqdm import tqdm
import fitsio
import lightkurve as lk

from fbpca import pca

from astropy.io import fits
import pandas as pd

from astropy.stats import sigma_clip
from astropy.time import Time


nast = 100
sector = 2
df = pd.read_csv("../../../catalog/first_big_catalog.csv.zip")
df = df[["pdes", "full_name", "max_Vmag", "sector"]][df.sector == b.sector].sort_values(
    "max_Vmag", ascending=False
)
df = df[(df.max_Vmag <= 20) & (df.max_Vmag != 0)][::-1].reset_index(drop=True)


eps = []
for adx, d in tqdm(
    df[:nast].iterrows(), total=nast, desc="Downloading asteroid ephemerides"
):
    ep = te.ephem(d.pdes, Time(time, format="jd")).reset_index()
    ep.time = Time(ep["time"]).value
    ep["pdes"] = d.pdes
    eps.append(ep)

eps = pd.concat(eps)


nb = 30
npca_components = 5
nsamp = 50

np1 = 2
clip_corners = False
nmask = 5


ccd = 3
camera = 1

# b = ah.BackgroundCorrector.load(f'sec{sector:04}_camera{camera}_ccd{ccd}.tessbkg_lite')

# time = np.zeros(len(b.fnames))
# for idx, fname in enumerate(tqdm(b.fnames)):
#     hdr = fitsio.read_header(fname)
#     time[idx] = (hdr["TSTART"] + hdr["TSTOP"]) / 2 + 2457000
# time = np.sort(time)

# dmed = b.dmed[:, b.dmed[0] != 1]
for pde in eps[(eps["ccd"] == ccd) & (eps["camera"] == camera)].pdes.unique():
    ep = eps[eps.pdes == pde]
    ep = ep[ep.ccd == ccd]

    col, row = np.asarray((np.round(ep.column) - 1), int), np.asarray(
        (np.round(ep.row) - 1), int
    )
    l = np.where(time == ep.time[0])[0][0]

    dat = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
    bkg = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
    Ts = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
    cols = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
    rows = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan

    l = np.where(time == ep.time[0])[0][0]

    for tdx in tqdm(np.arange(0, len(ep)), desc="Extracting asteroid from FFIs"):
        tmask = np.arange(np.max([tdx - nb, 0]), np.min([tdx + nb + 1, len(ep)]))
        with fitsio.FITS(b.fnames[tdx + l]) as fts:
            for c1 in np.arange(-np1, np1 + 1):
                for r1 in np.arange(-np1, np1 + 1):
                    if clip_corners:
                        if (np.abs(c1) == np1) & (np.abs(r1) == np1):
                            continue
                    c = col[tmask] + c1
                    r = row[tmask] + r1
                    for idx in range(len(c)):
                        if (
                            ((c[idx] - 45) < 0)
                            | ((r[idx]) < 0)
                            | ((c[idx] - 45 + 1) >= 2048)
                            | ((r[idx] + 1) >= 2048)
                        ):
                            continue
                        cols[r1 + np1, c1 + np1, tdx, tmask[idx]] = c[idx]
                        rows[r1 + np1, c1 + np1, tdx, tmask[idx]] = r[idx]
                        dat[r1 + np1, c1 + np1, tdx, tmask[idx]] = fts[1][
                            r[idx] : r[idx] + 1, c[idx] : c[idx] + 1
                        ]
                        Ts[r1 + np1, c1 + np1, tdx, tmask[idx]] = tdx + l
    # Masks out asteroid...
    tmask = np.ones(dat.shape, bool)
    for c1 in np.arange(-np1, np1 + 1):
        for r1 in np.arange(-np1, np1 + 1):
            for kdx in np.arange(-nmask, nmask + 1):
                tmask[r1 + np1, c1 + np1,] &= ~np.diag(
                    np.ones(len(ep), dtype=bool), k=kdx
                )[np.abs(kdx) :, np.abs(kdx) :]
    tmask &= np.isfinite(dat)

    bkg = np.zeros((np1 * 2 + 1, np1 * 2 + 1, len(ep), len(ep))) * np.nan
    coords = [
        (r + r1, c + c1)
        for c, r in zip(col, row)
        for c1 in np.arange(-np1, np1 + 1)
        for r1 in np.arange(-np1, np1 + 1)
    ]
    unq_coords = list(set(coords))
    # Xf = b.Xf[[np.ravel_multi_index(coord, (2048, 2048)) for coord in unq_coords]].toarray()

    locs = []
    for coord in unq_coords:
        if (coord[0] > 0) & (coord[0] < 2048) & (coord[1] > 0) & (coord[1] < 2048):
            locs.append(np.ravel_multi_index(coord, (2048, 2048)))
        else:
            locs.append(0)
    Xf = b.Xf[locs].toarray()
    bkg1 = Xf.dot(b.w[np.arange(len(ep)) + l].T)

    s = np.asarray([f"{l[0]},{l[1]}" for l in coords])
    i, j, k = np.unique(s, return_counts=True, return_inverse=True)
    bkg1 = bkg1[j]

    jdx = 0
    for idx, c, r in zip(range(len(col)), col, row):
        for c1 in np.arange(-np1, np1 + 1):
            for r1 in np.arange(-np1, np1 + 1):
                bkg[r1 + np1, c1 + np1, :, idx] = bkg1[jdx]
                jdx += 1
    bkg *= np.nan ** (~np.isfinite(dat))

    res = dat - bkg
    jitter_model = np.zeros_like(dat) * np.nan
    jitter_err = np.zeros_like(dat) * np.nan
    for tdx in tqdm(range(len(ep)), desc="Applying Correction"):
        t = Ts[np1, np1, :, tdx]
        k = np.isfinite(t)
        t = t[k]
        if len(t) == 0:
            continue
        X = np.hstack(
            [pca(dmed[t.astype(int), :], npca_components)[0], poly[t.astype(int)]]
        )
        j = tmask[np1, np1, k, tdx]

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

    # This time array might be obo...?
    k = np.isfinite(Ts[np1, np1, np.diag(np.ones(len(ep), dtype=bool))])
    lc = lk.LightCurve(
        time[Ts[np1, np1, np.diag(np.ones(len(ep), dtype=bool))][k].astype(int)],
        np.nansum(
            (dat - bkg - jitter_model)[:, :, np.diag(np.ones(len(ep), dtype=bool))][
                :, :, k
            ],
            axis=(0, 1),
        ),
        np.nansum(
            (jitter_err ** 2)[:, :, np.diag(np.ones(len(ep), dtype=bool))][:, :, k],
            axis=(0, 1),
        )
        ** 0.5,
        targetid=pde,
        label=pde,
    )
    lc.to_fits(
        f"/Users/ch/Projects/PDART/asteroid_lightcurves/fits/{pde}.fits",
        overwrite="True",
    )
