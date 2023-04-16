import numpy as np

from algorithms.objectives import (
    Objective,
    ObservedSurfaceMarginalObjective,
    ObservedSurfaceObjective,
    ObservedConfidenceLowerObjective,
    ObservedConfidenceUpperObjective,
    IntersectionOcclusionAwareObjective,
    IntersectionObjective,
    ConfidenceObjective,
    ConfidenceSimpleObjective,
    ConfidencePolarObjective,
    ConfidenceSimpleWeightedObjective,
    UncertaintyObjective,
    UncertaintyPolarObjective,
)
from algorithms.observations import Observations
from simulation.camera import Camera
from utils.math import is_in_range
from utils.tools import LazyDict

TRUE_ALGORITHM = "Greedy-ObservedSurfaceMarginal"


class Algorithm():
    """Base class for algorithms."""
    
    def __init__(self, gp=None):
        """Construct instance of algorithm.

        Args:
            gp (GaussianProcess, optional): Instance of Gaussian process model.
                Defaults to None.
        """
        self.observations = Observations()
        self.gp = gp
        self.linked = False

    def reset(self):
        """Reset knowledge of algorithm."""
        if self.linked:
            print("WARNING: not possible to reset linked algorithm")
            return
        self.observations.reset()
        self.gp.reset()

    def add_observation(self, observation, noise=0):
        """Update knowledge of algorithm through measurement.

        Args:
            observation (2xN array): Polar coordinates of measurements
            noise (scalar or N array, optional): Uniform noise if scalar or
                noise per measurement if array. Defaults to 0.
        """
        if self.linked:
            print("WARNING: not possible to add observations to linked algorithm")
            return
        self.observations.update(observation)
        self.gp.update(*observation, noise=noise)

    def compute_nbv(self):
        """Compute NBV based on knowledge of algorithm.

        Returns:
            float: Polar angle describing NBV.
        """        
        return 0
    
    @property
    def data(self):
        """Knowledge of algorithm obtained through previous measurements."""
        return {
            Objective.CONFIDENCE: self.gp,
            Objective.OBSERVATIONS: self.observations,
        }

    def link(self, algorithm):
        """Link knowledge of algorithm with another algorithm.
        
        This method is useful when simulating multiple algorithms simultaneously
        and allows knowledge propagation from a main algorithm to others. Linked
        algorithms are therefore not allowed to modify the linked knowledge of
        the main algorithm.

        Args:
            algorithm (Algorithm): Instance of algorithm.
        """
        self.observations = algorithm.observations
        self.gp = algorithm.gp
        self.linked = True


class GreedyAlgorithm(Algorithm):
    """Class for greedy algorithms maximizing some objective function."""
    
    def __init__(self, objective, n=100, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.thetas = np.linspace(0, 2*np.pi, num=n)

    def compute_nbv(self, with_estimates=False):
        """Compute NBV based on knowledge of algorithm.

        Args:
            with_estimates (bool, optional): Flag whether computed global
                estimates should be returned or not. Defaults to False.

        Returns:
            float or (float, N array): Polar coordinate of NBV and optionally
                list of number of estimated points from N camera locations.
        """
        # find NBV
        estimates = self.objective(self.thetas, self.data)
        nbv = self.thetas[np.argmax(estimates)]
        # return NBV and optionally estimates
        if not with_estimates:
            return nbv
        else:
            return nbv, estimates
    
    def compute_estimate_points(self, camera):
        """Compute points estimated to be observed from current camera location.

        This only works with numerical objective functions, while closed-form
        objective functions will print a warning.

        Args:
            camera (Camera): Instance of camera.

        Returns:
            2xN array: Polar coordinates of estimated points.
        """        
        return self.objective.compute_estimate_points(camera, self.data)


class TwoPhaseAlgorithm(Algorithm):
    """Class for greedy 2-phase algorithms.
    
    Phase 1: finding two boundaries maximizing objective 1
    Phase 2: finding NBV within boundaries maximizing objective 2.
    """

    def __init__(self, objective1, objective2, n=100, **kwargs):
        super().__init__(**kwargs)
        self.objective1 = objective1
        self.objective2 = objective2
        self.thetas = np.linspace(0, 2*np.pi, num=n)

    def compute_nbv(self, with_estimates=False):
        """Compute NBV based on knowledge of algorithm.

        Args:
            with_estimates (bool, optional): Flag whether computed global
                estimates should be returned or not. Defaults to False.

        Returns:
            float or (float, 2xN array): Polar coordinate of NBV and optionally
               list of number of estimated points from phase 1 objective and
               phase 2 objective from N camera locations.
        """
        # phase 1
        estimates1 = self.objective1(self.thetas, self.data)
        nbv1 = self.thetas[np.argmax(estimates1)]
        nbv1_phi1, nbv1_phi2 = self.objective1.get_summation_interval(Camera(nbv1), self.data)
        mask = is_in_range(self.thetas, (nbv1_phi1[0], nbv1_phi2[0]), mod=2*np.pi)
        # phase 2
        estimates2 = self.objective2(self.thetas, self.data)
        estimates2[np.logical_not(mask)] = np.nan
        nbv2 = self.thetas[np.nanargmax(estimates2)]
        # return NBV and optionally estimates
        if not with_estimates:
            return nbv2
        else:
            return nbv2, np.array([estimates1, estimates2])


def build_algorithms(build_gp=lambda: None, object=None):
    return LazyDict({
        # greedy algorithm, observation-based objective function
        "Greedy-ObservedSurface":         lambda: GreedyAlgorithm(ObservedSurfaceObjective(obj=object), gp=build_gp()),
        TRUE_ALGORITHM:                   lambda: GreedyAlgorithm(ObservedSurfaceMarginalObjective(obj=object), gp=build_gp()),
        "Greedy-ObservedConfidenceLower": lambda: GreedyAlgorithm(ObservedConfidenceLowerObjective(), gp=build_gp()),
        "Greedy-ObservedConfidenceUpper": lambda: GreedyAlgorithm(ObservedConfidenceUpperObjective(), gp=build_gp()),
        # greedy algorithm, intersection-based objective function
        "Greedy-IntersectionOcclusionAware": lambda: GreedyAlgorithm(IntersectionOcclusionAwareObjective(), gp=build_gp()),
        "Greedy-Intersection":               lambda: GreedyAlgorithm(IntersectionObjective(), gp=build_gp()),
        "Greedy-Intersection_cf":            lambda: GreedyAlgorithm(IntersectionObjective(use_cf=True), gp=build_gp()),
        # greedy algorithm, confidence-based objective function
        "Greedy-Confidence":                  lambda: GreedyAlgorithm(ConfidenceObjective(), gp=build_gp()),
        "Greedy-Confidence_cf":               lambda: GreedyAlgorithm(ConfidenceObjective(use_cf=True), gp=build_gp()),
        "Greedy-ConfidenceSimple":            lambda: GreedyAlgorithm(ConfidenceSimpleObjective(), gp=build_gp()),
        "Greedy-ConfidenceSimple_cf":         lambda: GreedyAlgorithm(ConfidenceSimpleObjective(use_cf=True), gp=build_gp()),
        "Greedy-ConfidenceSimplePolar":       lambda: GreedyAlgorithm(ConfidencePolarObjective(), gp=build_gp()),
        "Greedy-ConfidenceSimplePolar_cf":    lambda: GreedyAlgorithm(ConfidencePolarObjective(use_cf=True), gp=build_gp()),
        "Greedy-ConfidenceSimpleWeighted_cf": lambda: GreedyAlgorithm(ConfidenceSimpleWeightedObjective(use_cf=True), gp=build_gp()),
        # greedy algorithm, uncertainty-based objective function
        "Greedy-Uncertainty":         lambda: GreedyAlgorithm(UncertaintyObjective(), gp=build_gp()),
        "Greedy-Uncertainty_cf":      lambda: GreedyAlgorithm(UncertaintyObjective(use_cf=True), gp=build_gp()),
        "Greedy-UncertaintyPolar":    lambda: GreedyAlgorithm(UncertaintyPolarObjective(), gp=build_gp()),
        "Greedy-UncertaintyPolar_cf": lambda: GreedyAlgorithm(UncertaintyPolarObjective(use_cf=True), gp=build_gp()),
        # two-phase algorithm
        "TwoPhase-ConfidenceSimple-Uncertainty": lambda: TwoPhaseAlgorithm(ConfidenceSimpleObjective(), UncertaintyObjective(use_cf=True), gp=build_gp()),
    })

ALGORITHMS = list(build_algorithms().keys())
