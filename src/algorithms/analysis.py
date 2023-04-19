import numpy as np

import algorithms.gp as gp
import parameters as params
from simulation.camera import Camera

class Analysis():
    def __init__(self, delta=0.5, sigma_eps=params.OBS_NOISE, sigma_f=1, nu=2.5, l=1):
        # failure probability
        self.delta = delta
        # noise parameter
        self.sigma_eps = sigma_eps
        # kernel parameters
        self.sigma_f = sigma_f
        self.nu = nu
        self.l = l
        # kernel constants from Lipschitz-continuity assumption
        # (should be chosen depending on kernel parameters)
        self.a = 1          # no clue, should be related to union bound over domain
        self.b = np.sqrt(2) # guessed from expontial decaying bound on Gaussian-distributed random variables

    def beta_bound(self, t):
        # only valid as confidence parameter for nu > 2
        return 2 * np.log(2*np.pi**3*self.b/3 * np.sqrt(np.log(2*self.a/self.delta)) * t**4/self.delta)
    
    def gamma_bound_c4(self):
        k = gp.build_kernel_matern_periodic(sigma=1, l=self.l, nu=self.nu, normalized=False) # plain kernel without sigma_f and normalization
        c_nu = gp.eval_kernel(k, [[0]])
        A = np.power(self.sigma_f**2 / (c_nu*self.sigma_eps**2) * 2*np.sqrt(2*self.nu)*np.sinh(np.sqrt(2*self.nu)*np.pi / self.l) / (np.pi*self.l), 1/(2*self.nu+1))
        B = np.sqrt(4*np.pi**2 + self.l**2 / np.power(2, self.nu))
        return A*B

    def gamma_bound(self, t):
        # only valid for sigma_f = k_max = 1 (could be modified)
        c4 = self.gamma_bound_c4()
        return c4 * np.power(t, 1/(2*self.nu+1)) * np.power(np.log(1+t/(self.sigma_eps**2)), 2*self.nu/(2*self.nu+1)) + np.log(1+t/(self.sigma_eps**2))

    def cum_regret_bound_c1(self):
        fov_width = 2*np.arctan(np.sin(params.CAM_FOV_RAD()/2)*params.CAM_DOF / (params.CAM_D - np.cos(params.CAM_FOV_RAD()/2)*params.CAM_DOF))
        return 2*params.OBJ_D_MAX*fov_width / params.GRID_H**2 * np.sqrt(3 / np.log(params.OBS_NOISE**-2 + 1))
    
    def cum_regret_bound_c2(self):
        fov_width = 2*np.arctan(np.sin(params.CAM_FOV_RAD()/2)*params.CAM_DOF / (params.CAM_D - np.cos(params.CAM_FOV_RAD()/2)*params.CAM_DOF))
        return params.OBJ_D_MAX*fov_width / params.GRID_H**2 * np.pi / np.sqrt(3)

    def cum_regret_bound(self, t, with_constants=True):
        beta_bound_t = self.beta_bound(t)
        gamma_bound_t = self.gamma_bound(t)
        c1 = self.cum_regret_bound_c1()
        c2 = self.cum_regret_bound_c2()
        if with_constants:
            return c1 * np.sqrt(t * beta_bound_t * gamma_bound_t) + c2
        else:
            return np.sqrt(t * beta_bound_t * gamma_bound_t)
    
    def avg_regret_bound(self, t, with_constants=True):
        return self.cum_regret_bound(t, with_constants=with_constants) / t
