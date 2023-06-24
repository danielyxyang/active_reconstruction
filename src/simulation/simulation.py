import numpy as np

import parameters as params
from algorithms.gp import GaussianProcess, build_mean
from algorithms.algorithms import Algorithm, build_algorithms, ALGORITHMS, TRUE_ALGORITHM
from simulation.camera import Camera
from utils_ext.math import setdiff2d


class Simulation():
    def __init__(self, object, camera, algorithm):
        self.obj = object
        self.camera = camera
        self.algorithm = algorithm
        self.converged = False

        self.n_marginal = [] # store number of newly observed points
        self.n_marginal_opt = [] # store largest possible number of newly observed points
        self.algorithm_opt = build_algorithms(object=object)[TRUE_ALGORITHM]
        self.algorithm_opt.link(self.algorithm)

        self.reset()

    # LOW-LEVEL METHODS TO RUN SIMULATION

    def reset(self):
        """Reset state of simulation."""
        self.algorithm.reset()
        self.camera.observe(self.obj.surface_points)
        self.converged = False

        self.n_marginal = []
        self.n_marginal_opt = []

    def take_measurement(self):
        """Add current observations of camera to knowledge of algorithm."""
        # compute number of marginal observations
        n_marginal_observation = len(setdiff2d(self.camera.observation.T, self.algorithm.observations.observed_points.T))
        if n_marginal_observation == 0:
            self.converged = True
            return
        self.n_marginal.append(n_marginal_observation)

        # compute regret
        camera_opt = Camera(theta=self.algorithm_opt.compute_nbv())
        camera_opt.observe(self.obj.surface_points)
        n_marginal_observation_opt = len(setdiff2d(camera_opt.observation.T, self.algorithm.observations.observed_points.T))
        self.n_marginal_opt.append(n_marginal_observation_opt)

        # add observations to algorithm 
        self.algorithm.add_observation(self.camera.observation, noise=params.OBS_NOISE)

    def move_camera(self, theta):
        """Move camera to given position and update observations of camera."""
        self.camera.move(theta)
        self.camera.observe(self.obj.surface_points)

    def is_converged(self):
        """Return flag indicating convergence defined as no new observations."""
        return self.converged

    # HIGH-LEVEL METHODS TO RUN SIMULATION

    def step(self, nbv=None):
        """Move camera to NBV or given position and take measurement."""
        if nbv is None:
            nbv = self.algorithm.compute_nbv()
        self.move_camera(nbv)
        self.take_measurement()
    
    def run(self, thetas=None, n=None):
        """Run simulation with the given positions or n times the NBV."""
        if thetas is not None:
            for theta in thetas:
                self.step(nbv=theta)
        elif n is not None:
            for _ in range(n):
                self.step()
        nbv = self.algorithm.compute_nbv()
        self.move_camera(nbv)
        
    # METHODS TO EVALUATE SIMULATION

    def progress(self):
        """Return relative amount of reconstructed object."""
        return np.sum(self.n_marginal) / len(self.obj.surface_points.T)
    
    def results(self):
        """Return SimulationResults object for evaluating simulation."""
        return SimulationResults(self.n_marginal, self.n_marginal_opt, len(self.obj.surface_points.T))

    # STATIC METHODS

    @staticmethod
    def build(object, camera, kernel, algorithm=None):
        # build algorithm
        build_gp = lambda: GaussianProcess(build_mean(), kernel, discretize=True)
        if algorithm in ALGORITHMS:
            algorithm = build_algorithms(build_gp=build_gp, object=object)[algorithm]
        else:
            algorithm = Algorithm(gp=build_gp())
        # build simulation
        simulation = Simulation(object, camera, algorithm)
        return simulation


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
        """Number of measurements."""
        return len(self.n_marginal)
    def n_measurements_upto_thresh(self, thresh):
        """Number of measurements up to relative reconstruction above thresh."""
        return self._nan_if_empty(np.min, self.rounds[self.n_total_rel >= thresh])
    @property
    def rounds(self):
        """List of rounds."""
        return np.arange(1, self.n_measurements + 1)
    
    @property
    def n_total(self):
        """List of number of observed surface points."""
        return np.cumsum(self.n_marginal)
    @property
    def n_total_rel(self):
        """List of relative number of observed surface points."""
        return self.n_total / self.n_max
    @property
    def n_total_final(self):
        """Total number of observed surface points."""
        return self.n_total[-1] if self.n_measurements > 0 else 0
    @property
    def n_total_final_rel(self):
        """Relative number of observed surface points."""
        return self.n_total_final / self.n_max
    @property
    def n_remaining(self):
        """Total number of unobserved surface points."""
        return self.n_max - self.n_total_final
    
    @property
    def regret(self):
        """List of simple individual regret."""
        return self.n_marginal_opt - self.n_marginal
    @property
    def regret_avg(self):
        """Average of simple individual regret."""
        return self._nan_if_empty(np.mean, self.regret)
    @property
    def regret_max(self):
        """Maximum simple individual regret."""
        return self._nan_if_empty(np.max, self.regret)
    @property
    def regret_min(self):
        """Minimum simple individual regret."""
        return self._nan_if_empty(np.min, self.regret)

