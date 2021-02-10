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
        m |= (np.abs(x) > (std * sigma))
    return std


class Data(object):

    def __init__(self, time, data, error, spline_days=0.5, psf_components=5, bkg_components=5):
        self.time = time
        self.data = data
        self.error = error
        self.shape = data.shape
        Y, X = np.mgrid[:data.shape[1], :data.shape[2]]
        self.X = X.ravel()
        self.Y = Y.ravel()
        self.ntime = self.shape[0]
        self.npixel = np.product(self.shape[1:])
        self.med_frame = np.median(self.data, axis=0)
        self.mask = np.ones(self.data.shape, bool)
        self._make_A(psf_components=psf_components, bkg_components=bkg_components, spline_days=spline_days)
        self._find_outliers()
        self._make_A(psf_components=psf_components, bkg_components=bkg_components, spline_days=spline_days)
#        self.bkg = self._get_basic_bkg()
#        self.basic_res = self.data - self.bkg
#        self.med_frame = np.median(self.basic_res, axis=0)
#        stats = sigma_clipped_stats(self.basic_res, axis=(1, 2))
#        self.pixel_std = stats[2]

    def _make_A(self, psf_components, bkg_components, spline_days, spline=True):

        goodpix = self.med_frame > np.percentile(self.med_frame, 80)
        goodpix &= (~self.mask).sum(axis=0) < 3

        # Capture the PCA from focus change and velocity aberration
        U, s, V = pca(self.data[:, goodpix], k=psf_components, n_iter=10)

#        U_x = pca(np.hstack([U * U[:, idx][:, None] for idx in range(psf_components)]), k=psf_components//2, n_iter=10)[0]
#        U = np.hstack([U, U_x])
        # Capture the background (as best as possible...)
        goodpix = self.med_frame <= np.percentile(self.med_frame, 20)
        goodpix &= (~self.mask).sum(axis=0) < 3
        U2, s2, V2 = pca(self.data[:, goodpix], k=bkg_components, n_iter=10)

        # 0.5 day spline to get rid of stellar variability and other garbage.

        # A polynomial for the first 30 days
        #st = ((self.time - self.time[15])/(self.time[30] - self.time[0])) * (self.time < self.time[30])
        #small_poly = np.vstack([st**0, st, st**2]).T
        # Beautiful design matrix
        if spline:
            spline = lk.designmatrix.create_spline_matrix(self.time, n_knots=int((self.time.max() - self.time.min()) / spline_days)).append_constant().X
            self.A = np.hstack([U, U2, spline])
        else:
            self.A = np.hstack([U, U2, U2[:, -1][:, None]**0])

        # Some priors, these are pretty wide
        self.prior_sigma = np.ones(self.A.shape[1]) * 100
        self.prior_mu = np.zeros((self.A.shape[1]))
        #prior_sigma[-spline.shape[1]:] = 10

        # The prior for the mean. Note we're going to set this as we fit each pixel
        self.prior_sigma[-1] = 1
        self.prior_mu[-1] = 1


    def __repr__(self):
        return f'Data [{self.shape}]'

    def _get_basic_bkg(self, straps=False):
        a, b = (self.X - self.X.mean())/self.X.max(), (self.Y - self.Y.mean())/self.Y.max()
        poly = np.vstack([a**0,
                              a, b, a*b,
                              a**2*b, a*b**2, a**2*b**2, a**2, b**2,
                            ]).T
        nterms = poly.shape[1]
        prior_sigma = np.ones(nterms) * 5000
        prior_mu = np.zeros(nterms)

        if straps:
            lines = np.asarray([(a == a1).astype(float) for a1 in np.unique(a)]).T
            prior_sigma = np.hstack([prior_sigma, np.ones(lines.shape[1]) * 500])
            prior_mu = np.hstack([prior_mu, np.zeros(lines.shape[1])])
            poly = np.hstack([poly, lines])
            poly = sparse.csr_matrix(poly)

        bkgs = np.zeros((self.ntime, self.npixel))
        for tdx, f, e in zip(range(self.ntime), self.data, self.error):
            k = f.ravel() < np.percentile(f, 95)
            prior_mu[0] = np.median(f)
            sigma_w_inv = poly[k].T.dot(poly[k]/e.ravel()[k, None]**2)
            sigma_w_inv += np.diag(1/prior_sigma**2)
            B = poly[k].T.dot((f.ravel()/e.ravel()**2)[k])
            B += prior_mu/prior_sigma**2
            bkg_w = np.linalg.solve(sigma_w_inv, B)
            bkg = poly.dot(bkg_w)
            bkg -= -np.percentile(f.ravel() - (bkg), 1)
            bkgs[tdx] = bkg
        return bkgs.reshape(self.shape)

    def _find_outliers(self):
        mask = np.ones((self.ntime, self.npixel), bool)
        data = self.data.reshape((self.ntime, self.npixel))
        error = self.error.reshape((self.ntime, self.npixel))

        # One quick run through to find out which points to mask
        sigma_w_inv = self.A.T.dot(self.A)#.toarray()

        for idx in tqdm(range(self.npixel)):
            f = data[:, idx]
            e = error[:, idx]

            B = self.A.T.dot(f)
            m = self.A.dot(np.linalg.solve(sigma_w_inv, B))

            # Build residuals
            test = (f - m)

            # Identify outliers
#            std = _std_iter(test, (self.pixel_std > 10), sigma=3, n_iters=2)
            std = _std_iter(test, ~mask[:, idx], sigma=3, n_iters=2)
            test /= (e**2 + std**2)**0.5/2

            # Store them in the mask
            mask[:, idx] &= (test < 3)

        # Get rid of points that are next to bad points in time and space
        mask = mask.reshape(self.shape)
        grads = np.asarray(np.gradient(mask.astype(float)))
        self.mask = mask | np.all(grads == 0, axis=0)

    def _find_model(self):
        model = np.zeros((self.ntime, self.npixel))
        model_err = np.zeros((self.ntime, self.npixel))

        data = self.data.reshape((self.ntime, self.npixel))
        error = self.error.reshape((self.ntime, self.npixel))
        mask = self.mask.reshape((self.ntime, self.npixel))


        # Run through fitting the model, ignoring those masked points.
        for idx in tqdm(range(np.product(self.shape[1:]))):
            f = data[:, idx]
            e = error[:, idx]
            k = mask[:, idx]

            # Set the prior mean
            self.prior_mu[-1] = np.average(f, weights=1/e**2)
            self.prior_sigma[-1] = np.std(f)

            sigma_w_inv = self.A[k].T.dot(self.A[k]/e[k, None]**2)
            sigma_w_inv += np.diag(1/self.prior_sigma**2)
            B = self.A[k].T.dot(f[k]/e[k]**2)
            B += self.prior_mu/self.prior_sigma**2
            w = np.linalg.solve(sigma_w_inv, B)
            model[:, idx] = self.A.dot(w)
#            sigma_w = np.linalg.inv(sigma_w_inv)
#            samp = np.random.multivariate_normal(w, sigma_w, 30)
#            model_err[:, idx] = np.std(np.asarray([self.A.dot(samp[idx]) for idx in range(30)]), axis=0)
#            model_err[:, idx] = self.A.dot(np.linalg.inv(sigma_w_inv).diagonal()**0.5)

#            import pdb;pdb.set_trace()
#            model_err[:, idx] = self.A.dot(np.linalg.inv(sigma_w_inv).diagonal()**0.5)
            #model_err[:, idx] = self.A.dot(np.linalg.solve(sigma_w_inv, self.A.T)).diagonal()**0.5
            #break
        self.model = model.reshape(self.shape)
        self.model_err = model_err.reshape(self.shape)
