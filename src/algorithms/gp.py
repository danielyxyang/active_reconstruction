import numpy as np
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern

from parameters import OBJ_D_AVG
from utils.helpers import polar_to_cartesian, polar_to_pixel
from utils.profiler import Profiler


class GaussianProcess():
    # reference: https://peterroelants.github.io/posts/gaussian-process-tutorial/
    
    def __init__(self, mean_func, kernel_func, n_std=2, n=1000, discretize=False):
        self.mean_func = mean_func
        self.kernel_func = kernel_func
        self.n_std = n_std # number of standard deviations for confidence bound
        self.x_eval = np.linspace(0, 2*np.pi, n)
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
        # reset observations
        self.x = np.array([])
        self.y = np.array([])
        self.noise = np.array([])
        # compute prior distribution
        self.mean, self.cov = self.evaluate()
        # discretize confidence bounds
        if self.discretize:
            self.__discretize()

    def update(self, x1, y1, noise=0):
        if np.isscalar(noise):
            noise = np.full_like(x1, noise)
        # add observations
        self.x = np.concatenate([self.x, x1])
        self.y = np.concatenate([self.y, y1])
        self.noise = np.concatenate([self.noise, noise])
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
        noise = (self.noise ** 2) * np.eye(len(self.x))
        sigma11 = self.kernel_func(self.x[:, np.newaxis], self.x[:, np.newaxis]) + noise
        sigma12 = self.kernel_func(self.x[:, np.newaxis], x_eval[:, np.newaxis])
        sigma22 = self.kernel_func(x_eval[:, np.newaxis], x_eval[:, np.newaxis])
        reg = gamma * np.eye(len(self.x))
        solved = np.linalg.solve(sigma11 + reg, sigma12).T
        mean = solved @ (self.y - self.mean_func(self.x)) + self.mean_func(x_eval)
        cov = sigma22 - (solved @ sigma12)
        return mean, cov

    def confidence_boundary(self, x_eval=None, interp=False):
        """Compute polar coordinates of the confidence boundary."""
        # set mean and covariance
        if x_eval is None:
            x_eval = self.x_eval
            mean, cov = self.mean, self.cov
        else:
            if interp:
                mean, cov = self.mean, self.cov
            else:
                mean, cov = self.evaluate(x_eval)
        # compute upper and lower confidence bound
        std = np.sqrt(np.diag(cov))
        lower = mean - self.n_std * std
        upper = mean + self.n_std * std
        # interpolate upper and lower confidence bound if necessary
        if x_eval is not None and interp:
            lower = np.interp(x_eval, self.x_eval, lower, period=2*np.pi)
            upper = np.interp(x_eval, self.x_eval, upper, period=2*np.pi)
        return np.array([x_eval, lower]), np.array([x_eval, upper])

    def confidence_region(self):
        """Compute cartesian coordinates of polygon vertices describing the confidence region.
        
        Note: The returned polygon vertices are not "well-defined" along the positive
        x-axis to ensure it forms a single closed polygon."""
        lower, upper = self.confidence_boundary()
        lower_x, lower_y = polar_to_cartesian(*lower)
        upper_x, upper_y = polar_to_cartesian(*upper)
        return np.concatenate([
            (upper_x, upper_y),
            (lower_x[::-1], lower_y[::-1]),
        ], axis=1)

    # PRIVATE METHODS

    def __discretize(self):
        """Compute polar coordinates of points on lower and upper confidence bound in each pixel."""
        with self.profiler.cm("discretization (GP)"):
            lower, upper = self.confidence_boundary()
            _, lower_pixel_indices = np.unique(polar_to_pixel(*lower), return_index=True, axis=1)
            _, upper_pixel_indices = np.unique(polar_to_pixel(*upper), return_index=True, axis=1)
            lower_pixel_indices = np.sort(lower_pixel_indices)
            upper_pixel_indices = np.sort(upper_pixel_indices)
            self.lower_points = lower[:, lower_pixel_indices]
            self.upper_points = upper[:, upper_pixel_indices]
            
            # check if evaluation of GP is fine enough for discretization
            n_lower_points = len(self.lower_points.T)
            n_upper_points = len(self.upper_points.T)
            n_samples = len(self.x_eval)
            if max(n_lower_points, n_upper_points) > 0.5 * n_samples:
                print("WARNING: consider increasing number of samples for discretizing GP confidence bounds ({} lower pixels, {} upper pixels, {} discretization samples)".format(n_lower_points, n_upper_points, n_samples))
    

def build_mean():
    return lambda phi: np.full_like(phi, fill_value=OBJ_D_AVG, dtype=float)


def build_kernel_rbf(sigma=1, l=1):
    k = RBF(length_scale=l)
    def kernel(x1, x2):
        return sigma**2 * k(x1, x2)
    return kernel


def build_kernel_periodic(sigma=1, l=1):
    k = ExpSineSquared(length_scale=l, periodicity=2*np.pi)
    def kernel(x1, x2):
        return sigma**2 * k(x1, x2)
    return kernel


def build_kernel_matern(sigma=1, l=1, nu=1.5):
    k = Matern(length_scale=l, nu=nu)
    def kernel(x1, x2):
        return sigma**2 * k(x1, x2)
    return kernel


def build_kernel_matern_periodic(sigma=1, l=1, nu=1.5, n_approx=1):
    k = Matern(length_scale=l, nu=nu)
    def kernel(x1, x2):
        ks = [k(x1 + i * 2*np.pi, x2) for i in range(-n_approx, n_approx+1)]
        return sigma**2 * np.sum(ks, axis=0)
    return kernel
