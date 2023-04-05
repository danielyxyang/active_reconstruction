import numpy as np

import parameters as params
from algorithms.algorithms import GreedyAlgorithm
from algorithms.objectives import ObservedSurfaceMarginalObjective
from simulation.camera import Camera
from utils.math import setdiff2d


class Simulation():
    def __init__(self, object, camera, algorithm):
        self.obj = object
        self.camera = camera
        self.algorithm = algorithm
        self.converged = False

        self.n_marginal = None # store marginal number of observed points for each measurement
        self.regret = None
        self.algorithm_opt = GreedyAlgorithm(ObservedSurfaceMarginalObjective(obj=object), gp=algorithm.gp)

        self.reset()

    # METHODS TO RUN SIMULATION

    def reset(self):
        self.algorithm.reset()
        self.camera.observe(self.obj.surface_points)
        self.converged = False

        self.n_marginal = []
        self.regret = []

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
        # observation_opt = Camera(theta=self.algorithm_opt.compute_nbv()).compute_observation(self.obj.surface_points)
        n_marginal_observation_opt = len(setdiff2d(camera_opt.observation.T, self.algorithm.observations.T))
        self.regret.append(n_marginal_observation_opt - n_marginal_observation)

        self.algorithm.add_observation(self.camera.observation, noise=params.GRID_H)

    def move_camera(self, theta):
        self.camera.move(theta)
        self.camera.observe(self.obj.surface_points)

    def step(self):
        nbv = self.algorithm.compute_nbv()
        self.move_camera(nbv)
        self.take_measurement()

    def is_converged(self):
        return self.converged
        
    # METHODS TO EVALUATE SIMULATION

    def progress(self):
        return np.sum(self.n_marginal) / len(self.obj.surface_points.T)
    
    def results(self):
        n_measurements = len(self.n_marginal)
        rounds = np.arange(1, n_measurements + 1)
        n_marginal = self.n_marginal
        n_total = np.cumsum(n_marginal)
        n_max = len(self.obj.surface_points.T)
        n_remaining = n_max - n_total[-1] if n_measurements > 0 else n_max
        regret = self.regret
        return {
            "n_measurements": n_measurements,
            "n_marginal": np.array([rounds, n_marginal]),
            "n_total": np.array([rounds, n_total / n_max]),
            "n_remaining": n_remaining,
            "regret": np.array([rounds, regret]),
        }
