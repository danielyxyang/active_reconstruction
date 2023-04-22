import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern, _check_length_scale

import parameters as params
from utils.math import polar_to_cartesian, polar_to_pixel
from utils.tools import Profiler


class GaussianProcess():
    def __init__(self, mean_func, kernel_func, x_eval=None, period=2*np.pi, n_std=2, n_dims=1, discretize=False):
        self.mean_func = mean_func
        self.kernel_func = kernel_func
        self.x_eval = x_eval if x_eval is not None else np.linspace(0, 2*np.pi, 1000)
        self.period = period
        self.n_std = n_std # number of standard deviations for confidence bound
        self.n_dims = n_dims # number of feature dimensions
        
        self.x = None
        self.y = None
        self.noise = None
        self.mean = None
        self.cov = None

        self.discretize = discretize
        self.lower_points = np.empty((2, 0))
        self.upper_points = np.empty((2, 0))
        self.profiler = Profiler()

        self.reset()

    def reset(self):
        """Reset Gaussian process to prior."""
        # reset observations
        self.x = np.empty(0) if self.n_dims == 1 else np.empty((0, self.n_dims))
        self.y = np.empty(0)
        self.noise = np.empty(0)
        # compute prior distribution
        self.mean, self.cov = self.evaluate()
        # discretize confidence bounds
        if self.discretize:
            self.__discretize()

    def update(self, x1, y1, noise=0):
        """Update Gaussian process with observations to posterior."""
        if np.isscalar(noise):
            noise = np.full(len(x1), noise)
        # add observations
        self.x = np.append(self.x, x1, axis=0)
        self.y = np.append(self.y, y1)
        self.noise = np.append(self.noise, noise)
        # compute posterior distribution
        self.mean, self.cov = self.evaluate()
        # discretize confidence bounds
        if self.discretize:
            self.__discretize()

    def evaluate(self, x_eval=None, gamma=1e-8):
        """Compute Gaussian distribution for given evaluation points."""
        if x_eval is None:
            x_eval = self.x_eval
        # compute posterior mean and covariance
        # reference: https://peterroelants.github.io/posts/gaussian-process-tutorial/
        noise = (self.noise ** 2) * np.eye(len(self.x))
        sigma11 = self.kernel_func(self.x, self.x) + noise
        sigma12 = self.kernel_func(self.x, x_eval)
        sigma22 = self.kernel_func(x_eval, x_eval)
        reg = gamma * np.eye(len(self.x))
        solved = np.linalg.solve(sigma11 + reg, sigma12).T
        mean = self.mean_func(x_eval) + solved @ (self.y - self.mean_func(self.x))
        cov = sigma22 - solved @ sigma12
        return mean, cov
    
    def sample(self, x_eval=None, interp=False, n=1):
        """Sample functions from Gaussian process at given evaluation points."""
        # set mean and covariance
        if x_eval is None or interp:
            mean, cov = self.mean, self.cov
        else:
            mean, cov = self.evaluate(x_eval)
        # sample functions
        samples = np.random.multivariate_normal(mean, cov, size=n)
        # interpolate sample functions if necessary
        if interp:
            samples = np.array([
                np.interp(x_eval, self.x_eval, samples[i], period=self.period)
                for i in range(n)
            ])
        return samples

    def confidence_boundary(self, x_eval=None, interp=False):
        """Compute polar coordinates of the confidence boundary."""
        # set mean and covariance
        if x_eval is None or interp:
            mean, cov = self.mean, self.cov
        else:
            mean, cov = self.evaluate(x_eval)
        # compute upper and lower confidence bound
        std = np.sqrt(np.diag(cov))
        lower = mean - self.n_std * std
        upper = mean + self.n_std * std
        # interpolate upper and lower confidence bound if necessary
        if interp:
            lower = np.interp(x_eval, self.x_eval, lower, period=self.period)
            upper = np.interp(x_eval, self.x_eval, upper, period=self.period)
        return lower, upper

    def confidence_region(self):
        """Compute cartesian coordinates of polygon vertices describing the confidence region.
        
        Note: The returned polygon vertices are not "well-defined" along the positive
        x-axis to ensure it forms a single closed polygon.
        """
        lower, upper = self.confidence_boundary()
        lower_x, lower_y = polar_to_cartesian(self.x_eval, lower)
        upper_x, upper_y = polar_to_cartesian(self.x_eval, upper)
        return np.concatenate([
            (upper_x, upper_y),
            (lower_x[::-1], lower_y[::-1]),
        ], axis=1)

    # PRIVATE METHODS

    def __discretize(self):
        """Compute polar coordinates of points on lower and upper confidence bound in each pixel."""
        with self.profiler.cm("discretization (GP)"):
            lower, upper = self.confidence_boundary()
            _, lower_pixel_indices = np.unique(polar_to_pixel(self.x_eval, lower), return_index=True, axis=1)
            _, upper_pixel_indices = np.unique(polar_to_pixel(self.x_eval, upper), return_index=True, axis=1)
            lower_pixel_indices = np.sort(lower_pixel_indices)
            upper_pixel_indices = np.sort(upper_pixel_indices)
            self.lower_points = np.array([self.x_eval, lower])[:, lower_pixel_indices]
            self.upper_points = np.array([self.x_eval, upper])[:, upper_pixel_indices]
            
            # check if evaluation of GP is fine enough for discretization (mostly just required for interpolation)
            n_lower_points = len(self.lower_points.T)
            n_upper_points = len(self.upper_points.T)
            n_samples = len(self.x_eval)
            if max(n_lower_points, n_upper_points) > 0.8 * n_samples:
                print("WARNING: consider increasing number of samples for discretizing GP confidence bounds ({} lower pixels, {} upper pixels, {} discretization samples)".format(n_lower_points, n_upper_points, n_samples))
    

def build_mean():
    return lambda phi: np.full_like(phi, fill_value=params.OBJ_D_AVG(), dtype=float)


def build_kernel_rbf(sigma=1, l=1):
    k = RBF(length_scale=l)
    def kernel(X1, X2):
        X1, X2 = _atleast_2d(X1, X2)
        return sigma**2 * k(X1, X2)
    return kernel


def build_kernel_rbf_periodic(sigma=1, l=1):
    k = ExpSineSquared(length_scale=l, periodicity=2*np.pi)
    def kernel(X1, X2):
        X1, X2 = _atleast_2d(X1, X2)
        return sigma**2 * k(X1, X2)
    return kernel


def build_kernel_matern(sigma=1, l=1, nu=1.5):
    k = Matern(length_scale=l, nu=nu)
    def kernel(X1, X2):
        X1, X2 = _atleast_2d(X1, X2)
        return sigma**2 * k(X1, X2)
    return kernel


def build_kernel_matern_periodic(sigma=1, l=1, nu=1.5, normalized=True):
    if nu == 0.5:
        def kernel(X1, X2):
            X1, X2 = _atleast_2d(X1, X2)
            if np.shape(X1)[1] > 1 or np.shape(X2)[1] > 1:
                print("WARNING: not implemented to evaluate periodic Matérn kernel on non-scalar samples!") # TODO
            # normalize input to [0,2pi)
            X1 = np.asarray(X1) % (2*np.pi)
            X2 = np.asarray(X2) % (2*np.pi)
            # evaluate closed-form kernel
            R = _compute_distances(X1, X2, l)
            u = R - np.pi/l
            return np.cosh(u)
        c = kernel([[0]], [[0]]) if normalized else 1 # normalization constant
        return lambda X1, X2: sigma**2/c * kernel(X1, X2)
    elif nu == 1.5:
        def kernel(X1, X2):
            X1, X2 = _atleast_2d(X1, X2)
            if np.shape(X1)[1] > 1 or np.shape(X2)[1] > 1:
                print("WARNING: not implemented to evaluate periodic Matérn kernel on non-scalar samples!") # TODO
            # normalize input to [0,2pi)
            X1 = np.asarray(X1) % (2*np.pi)
            X2 = np.asarray(X2) % (2*np.pi)
            # evaluate closed-form kernel
            R = _compute_distances(X1, X2, l)
            u = np.sqrt(3) * (R - np.pi/l)
            a0 = np.pi*l/6 * (l/np.pi + np.sqrt(3)/np.tanh(np.sqrt(3)*np.pi/l))
            a1 = -l**2/6
            return a0 * np.cosh(u) + a1 * u*np.sinh(u)
        c = kernel([[0]], [[0]]) if normalized else 1 # normalization constant
        return lambda X1, X2: sigma**2/c * kernel(X1, X2)
    elif nu == 2.5:
        def kernel(X1, X2):
            X1, X2 = _atleast_2d(X1, X2)
            if np.shape(X1)[1] > 1 or np.shape(X2)[1] > 1:
                print("WARNING: not implemented to evaluate periodic Matérn kernel on non-scalar samples!") # TODO
            # normalize input to [0,2pi)
            X1 = np.asarray(X1) % (2*np.pi)
            X2 = np.asarray(X2) % (2*np.pi)
            # evaluate closed-form kernel
            R = _compute_distances(X1, X2, l)
            u = np.sqrt(5) * (R - np.pi/l)
            a0 = (np.pi*l)**2/200 * (-5 + 3*(l/np.pi)**2 + 3*np.sqrt(5)*(l/np.pi)/np.tanh(np.sqrt(5)*np.pi/l) + 10/np.tanh(np.sqrt(5)*np.pi/l)**2)
            a1 = -np.pi*l**3/100 * (3/2*l/np.pi + np.sqrt(5)/np.tanh(np.sqrt(5)*np.pi/l))
            a2 = l**4/200
            return a0 * np.cosh(u) + a1 * u*np.sinh(u) + a2 * u**2*np.cosh(u)
        c = kernel([[0]], [[0]]) if normalized else 1 # normalization constant
        return lambda X1, X2: sigma**2/c * kernel(X1, X2)
    elif nu % 1 == 0.5:
        # TODO
        print("WARNING: not implemented to evaluate periodic Matérn kernel with nu={}".format(nu))
    else:
        print("WARNING: no closed-form expression for periodic Matérn kernel with nu={} known".format(nu))


def build_kernel_matern_periodic_approx(sigma=1, l=1, nu=1.5, n_approx=1):
    k = Matern(length_scale=l, nu=nu)
    def kernel(X1, X2):
        X1, X2 = _atleast_2d(X1, X2)
        ks = [k(X1 + i * 2*np.pi, X2) for i in range(-n_approx, n_approx+1)]
        return sigma**2 * np.sum(ks, axis=0)
    return kernel


def build_kernel_matern_periodic_truncated(sigma=1, l=1, nu=1.5, c1=np.pi, c2=2*np.pi, n=None):
    if n is None:
        n = int(np.ceil(c2 / (2 * np.pi)))
    k = Matern(length_scale=l, nu=nu)
    # define truncated kernel
    t = _build_truncation_function(c1, c2)
    def k_trunc(X1, X2):
        X1, X2 = _atleast_2d(X1, X2)
        R = _compute_distances(X1, X2, l)
        return t(R) * k(X1, X2)
    # define periodic kernel
    def kernel(X1, X2):
        ks = [k_trunc(X1 + i * 2*np.pi, X2) for i in range(-n, n+1)]
        return sigma**2 * np.sum(ks, axis=0)
    return kernel


def build_kernel_matern_periodic_warped(sigma=1, l=1, nu=1.5):
    k = Matern(length_scale=l, nu=nu)
    # define warping function
    u = lambda x: np.concatenate([np.cos(x), np.sin(x)], axis=-1)
    # define periodic function
    def kernel(X1, X2):
        X1, X2 = _atleast_2d(X1, X2)
        if np.shape(X1)[1] > 1 or np.shape(X2)[1] > 1:
            print("WARNING: Matérn kernel periodized by warping might not work for non-scalar samples!") # TODO check?
        return sigma**2 * k(u(X1), u(X2))
    return kernel


def eval_kernel(k, r):
    return np.diag(k(r, np.zeros_like(r)))


# HELPER FUNCTIONS

def _atleast_2d(*arrays):
    """View inputs as arrays with at least two dimensions.

    Same as np.atleast_2d, but appends instead of prepends new axis to
    1-dimensional arrays.
    
    Reference: numpy.core.shape_base.py::atleast_2d
    """
    res = []
    for array in arrays:
        array = np.asanyarray(array)
        if array.ndim == 0:
            result = array.reshape(1, 1)
        elif array.ndim == 1:
            result = array[:, np.newaxis] # changed
        else:
            result = array
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def _compute_distances(X1, X2, l):
    """Compute matrix with pairwise distances between X1 and X2.

    Reference: sklearn.gaussian_process.kernels.py::Matern.__call__
    
    Args:
        X1: ndarray of shape (n_samples, n_features)
        X2: ndarray of shape (n_samples, n_features)
        l: length scale of the kernel
    """
    l = _check_length_scale(X1, l)
    R = cdist(X1 / l, X2 / l, metric="euclidean")
    return R


def _build_truncation_function(c1, c2):
    """Build function smoothly transitioning from 1 to 0 between [-c2,-c1] and [c1, c2]"""
    dc = c2 - c1
    safe_inv = lambda x, where: np.divide(1, x, where=where, out=np.full_like(x, np.nan, dtype=float))
    omega = lambda x: np.where(x > 1e-3, np.exp(-safe_inv(x, where=x > 1e-3)), 0) # since exp(-inv(1e-3)) = 0
    t = lambda x: omega((c2 - np.abs(x)) / dc) / (omega((c2 - np.abs(x)) / dc) + omega((np.abs(x) - c1) / dc))
    return t
