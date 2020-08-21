import numpy as np
from scipy import sparse
from tqdm.notebook import tqdm
import lightkurve as lk
from astropy.stats import sigma_clipped_stats, sigma_clip

from fbpca import pca

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
        m &= (np.abs(x) > (std * sigma))
    return std


class Data(object):

    def __init__(self, time, data, error, spline_days=0.5, psf_components=10, bkg_components=20):
        self.time = time
        self.data = data
        self.error = error
        self.shape = data.shape
        Y, X = np.mgrid[:data.shape[1], :data.shape[2]]
        self.X = X.ravel()
        self.Y = Y.ravel()
        self.ntime = self.shape[0]
        self.npixel = np.product(self.shape[1:])
        self.bkg = self._get_basic_bkg()
        self.basic_res = self.data - self.bkg
        self.med_frame = np.median(self.basic_res, axis=0)
        stats = sigma_clipped_stats(self.basic_res, axis=(1, 2))
        self.pixel_std = stats[2]

        # Capture the PCA from focus change and velocity aberration
        U, s, V = pca(self.basic_res[:, self.med_frame > 100], k=psf_components, n_iter=10)
        U_x = pca(np.hstack([U * U[:, idx][:, None] for idx in range(10)]), k=psf_components//2, n_iter=10)[0]
        U = np.hstack([U, U_x])
        # Capture the background (as best as possible...)
        U2, s2, V2 = pca(self.data[:, self.med_frame <= 100], k=bkg_components, n_iter=10)
        # 0.5 day spline to get rid of stellar variability and other garbage.
        spline = lk.designmatrix.create_spline_matrix(self.time, n_knots=int((self.time.max() - self.time.min()) / spline_days)).append_constant().X
        # Beautiful design matrix
        self.A = np.hstack([U, U2, spline])


        # Some priors, these are pretty wide
        self.prior_sigma = np.ones(self.A.shape[1]) * 100
        self.prior_mu = np.zeros((self.A.shape[1]))
        #prior_sigma[-spline.shape[1]:] = 10

        # The prior for the mean. Note we're going to set this as we fit each pixel
        self.prior_sigma[-1] = 100
        self.prior_mu[-1] = 1


    def __repr__(self):
        return f'Data [{self.shape}]'

    def _get_basic_bkg(self):
        a, b = (self.X - self.X.mean())/self.X.max(), (self.Y - self.Y.mean())/self.Y.max()
        poly = np.vstack([a**0,
                              a, b, a*b,
                              a**2*b, a*b**2, a**2*b**2, a**2, b**2,
                            ]).T
        nterms = poly.shape[1]
        prior_sigma = np.ones(nterms) * 5000
        prior_mu = np.zeros(nterms)

        bkgs = np.zeros((self.ntime, self.npixel))
        for tdx, f, e in zip(range(self.ntime), self.data, self.error):
            k = f.ravel() < np.percentile(f, 95)
            sigma_w_inv = poly[k].T.dot(poly[k]/e.ravel()[k, None]**2)
            sigma_w_inv += np.diag(1/prior_sigma**2)
            B = poly[k].T.dot((f.ravel()/e.ravel()**2)[k])
            B += prior_mu/prior_sigma**2
            bkg_w = np.linalg.solve(sigma_w_inv, B)
            bkg = poly.dot(bkg_w)
            bkg -= -np.percentile(f.ravel() - (bkg), 1)
            bkgs[tdx] = bkg
        return bkgs.reshape(tpf.flux.shape)


    def _find_outliers(self):
        mask = np.ones((self.ntime, self.npixel), bool)
        data = self.data.reshape((self.ntime, self.npixel))
        error = self.error.reshape((self.ntime, self.npixel))

        # One quick run through to find out which points to mask
        sigma_w_inv = self.A.T.dot(self.A)
        for idx in tqdm(range(self.npixel)):
            f = data[:, idx]
            e = error[:, idx]

            B = self.A.T.dot(f)
            m = self.A.dot(np.linalg.solve(sigma_w_inv, B))

            # Build residuals
            test = (f - m)

            # Identify outliers
            std = _std_iter(test, (self.pixel_std > 10), sigma=3, n_iters=2)
            test /= (e**2 + std**2)**0.5/2

            # Store them in the mask
            mask[:, idx] &= (test < 3)

        # Get rid of points that are next to bad points in time and space
        mask = mask.reshape(self.shape)
        grads = np.asarray(np.gradient(mask.astype(float)))
        self.mask = np.all(grads == 0, axis=0)

    def _find_model(self):
        model = np.zeros((self.ntime, self.npixel))
        model_err = np.zeros((self.ntime, self.npixel))

        data = self.data.reshape((self.ntime, self.npixel))
        error = self.error.reshape((self.ntime, self.npixel))
        mask = self.mask.reshape((self.ntime, self.npixel))


        # Run through fitting the model, ignoring those masked points.
        for idx in tqdm(range(np.product(tpf.flux.shape[1:]))):
            f = data[:, idx]
            e = error[:, idx]
            k = mask[:, idx]

            # Set the prior mean
            self.prior_mu[-1] = np.average(f, weights=1/e**2)

            sigma_w_inv = self.A[k].T.dot(self.A[k]/e[k, None]**2)
            sigma_w_inv += np.diag(1/self.prior_sigma**2)
            B = self.A[k].T.dot(f[k]/e[k]**2)
            B += self.prior_mu/self.prior_sigma
            w = np.linalg.solve(sigma_w_inv, B)
            model[:, idx] = self.A.dot(w)
            model_err[:, idx] = (self.A.dot(np.linalg.solve(sigma_w_inv, self.A.T)).diagonal()**0.5)

        self.model = model
        self.model_err = model_err
