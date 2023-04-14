import numpy as np

import parameters as params
from algorithms.algorithms import build_algorithms, TRUE_ALGORITHM
from simulation.camera import Camera
from utils.math import setdiff2d


class Simulation():
    def __init__(self, object, camera, algorithm):
        self.obj = object
        self.camera = camera
        self.algorithm = algorithm
        self.converged = False

        self.n_marginal = [] # store number of newly observed points
        self.n_marginal_opt = [] # store largest possible number of newly observed points
        self.algorithm_opt = build_algorithms(build_gp=lambda: algorithm.gp, object=object)[TRUE_ALGORITHM]["algorithm"]

        self.reset()

    # METHODS TO RUN SIMULATION

    def reset(self):
        self.algorithm.reset()
        self.camera.observe(self.obj.surface_points)
        self.converged = False

        self.n_marginal = []
        self.n_marginal_opt = []

    def take_measurement(self):
        # compute number of marginal observations
        n_marginal_observation = len(setdiff2d(self.camera.observation.T, self.algorithm.observations.T))
        if n_marginal_observation == 0:
            self.converged = True
            return
        self.n_marginal.append(n_marginal_observation)

        # compute regret
        self.algorithm_opt.reset(algorithm=self.algorithm)
        camera_opt = Camera(theta=self.algorithm_opt.compute_nbv())
        camera_opt.observe(self.obj.surface_points)
        n_marginal_observation_opt = len(setdiff2d(camera_opt.observation.T, self.algorithm.observations.T))
        self.n_marginal_opt.append(n_marginal_observation_opt)

        self.algorithm.add_observation(self.camera.observation, noise=params.OBS_NOISE)

    def move_camera(self, theta):
        self.camera.move(theta)
        self.camera.observe(self.obj.surface_points)

    def step(self, nbv=None):
        if nbv is None:
            nbv = self.algorithm.compute_nbv()
        self.move_camera(nbv)
        self.take_measurement()

    def is_converged(self):
        return self.converged
        
    # METHODS TO EVALUATE SIMULATION

    def progress(self):
        return np.sum(self.n_marginal) / len(self.obj.surface_points.T)
    
    def results(self):
        return SimulationResults(self.n_marginal, self.n_marginal_opt, len(self.obj.surface_points.T))


class SimulationResults():
    def __init__(self, n_marginal, n_marginal_opt, n_max):
        self.n_marginal = np.asarray(n_marginal)
        self.n_marginal_opt = np.asarray(n_marginal_opt)
        self.n_max = n_max
    
    def to_dict(self):
        return {
            "n_measurements": self.n_marginal,
            "n_measurements_opt": self.n_marginal_opt,
            "n_max": self.n_max,
        }
    @staticmethod
    def from_dict(dict):
        return SimulationResults(
            dict["n_measurements"],
            dict["n_measurements_opt"],
            dict["n_max"],
        )
    
    def _nan_if_empty(self, op, list):
        return op(list) if len(list) > 0 else np.nan

    @property
    def n_measurements(self):
        return len(self.n_marginal)
    def n_measurements_upto_thresh(self, thresh):
        return self._nan_if_empty(np.min, self.rounds[self.n_total_rel >= thresh])
    @property
    def rounds(self):
        return np.arange(1, self.n_measurements + 1)
    
    @property
    def n_total(self):
        return np.cumsum(self.n_marginal)
    @property
    def n_total_rel(self):
        return self.n_total / self.n_max
    @property
    def n_total_final(self):
        return self.n_total[-1] if self.n_measurements > 0 else 0
    @property
    def n_total_final_rel(self):
        return self.n_total_final / self.n_max
    @property
    def n_remaining(self):
        return self.n_max - self.n_total_final
    
    @property
    def regret(self):
        return self.n_marginal_opt - self.n_marginal
    @property
    def regret_avg(self):
        return self._nan_if_empty(np.mean, self.regret)
    @property
    def regret_max(self):
        return self._nan_if_empty(np.max, self.regret)
    @property
    def regret_min(self):
        return self._nan_if_empty(np.min, self.regret)

