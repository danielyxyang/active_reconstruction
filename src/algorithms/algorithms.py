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

TRUE_ALGORITHM = "Greedy-ObservedSurfaceMarginal"


class Algorithm():
    """Base class for algorithms."""
    
    def __init__(self, gp=None):
        self.observations = Observations()
        self.gp = gp
        self.linked = False

    def reset(self):
        if self.linked:
            print("WARNING: not possible to reset linked algorithm")
            return
        self.observations.reset()
        self.gp.reset()

    def add_observation(self, observation, noise=0):
        if self.linked:
            print("WARNING: not possible to add observations to linked algorithm")
            return
        self.observations.update(observation)
        self.gp.update(*observation, noise=noise)

    def compute_nbv(self):
        return 0
    
    @property
    def data(self):
        return {
            Objective.CONFIDENCE: self.gp,
            Objective.OBSERVATIONS: self.observations,
        }

    # HELPER METHODS

    def link(self, algorithm):
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
        # find NBV
        estimates = self.objective(self.thetas, self.data)
        nbv = self.thetas[np.argmax(estimates)]
        # return NBV and optionally estimates
        if not with_estimates:
            return nbv
        else:
            return nbv, estimates
    
    def compute_estimate_points(self, camera):
        return self.objective.compute_estimate_points(camera, self.data)


class TwoPhaseAlgorithm(Algorithm):
    """Class for greedy 2-phase algorithms.
    
    Phase 1: finding two boundaries maximizing objective 1
    Phase 2: finding NBV within boundaries maximizing objective 2."""

    def __init__(self, objective1, objective2, n=100, **kwargs):
        super().__init__(**kwargs)
        self.objective1 = objective1
        self.objective2 = objective2
        self.thetas = np.linspace(0, 2*np.pi, num=n)

    def compute_nbv(self, with_estimates=False):
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
    return {
        # greedy algorithm, observation-based objective function
        TRUE_ALGORITHM:                   GreedyAlgorithm(ObservedSurfaceMarginalObjective(obj=object), gp=build_gp()),
        "Greedy-ObservedSurface":         GreedyAlgorithm(ObservedSurfaceObjective(obj=object), gp=build_gp()),
        "Greedy-ObservedConfidenceLower": GreedyAlgorithm(ObservedConfidenceLowerObjective(), gp=build_gp()),
        "Greedy-ObservedConfidenceUpper": GreedyAlgorithm(ObservedConfidenceUpperObjective(), gp=build_gp()),
        # greedy algorithm, intersection-based objective function
        "Greedy-IntersectionOcclusionAware": GreedyAlgorithm(IntersectionOcclusionAwareObjective(), gp=build_gp()),
        "Greedy-Intersection":               GreedyAlgorithm(IntersectionObjective(), gp=build_gp()),
        "Greedy-Intersection_cf":            GreedyAlgorithm(IntersectionObjective(use_cf=True), gp=build_gp()),
        # greedy algorithm, confidence-based objective function
        "Greedy-Confidence":                  GreedyAlgorithm(ConfidenceObjective(), gp=build_gp()),
        "Greedy-Confidence_cf":               GreedyAlgorithm(ConfidenceObjective(use_cf=True), gp=build_gp()),
        "Greedy-ConfidenceSimple":            GreedyAlgorithm(ConfidenceSimpleObjective(), gp=build_gp()),
        "Greedy-ConfidenceSimple_cf":         GreedyAlgorithm(ConfidenceSimpleObjective(use_cf=True), gp=build_gp()),
        "Greedy-ConfidenceSimplePolar":       GreedyAlgorithm(ConfidencePolarObjective(), gp=build_gp()),
        "Greedy-ConfidenceSimplePolar_cf":    GreedyAlgorithm(ConfidencePolarObjective(use_cf=True), gp=build_gp()),
        "Greedy-ConfidenceSimpleWeighted_cf": GreedyAlgorithm(ConfidenceSimpleWeightedObjective(use_cf=True), gp=build_gp()),
        # greedy algorithm, uncertainty-based objective function
        "Greedy-Uncertainty":         GreedyAlgorithm(UncertaintyObjective(), gp=build_gp()),
        "Greedy-Uncertainty_cf":      GreedyAlgorithm(UncertaintyObjective(use_cf=True), gp=build_gp()),
        "Greedy-UncertaintyPolar":    GreedyAlgorithm(UncertaintyPolarObjective(), gp=build_gp()),
        "Greedy-UncertaintyPolar_cf": GreedyAlgorithm(UncertaintyPolarObjective(use_cf=True), gp=build_gp()),
        # two-phase algorithm
        "TwoPhase-ConfidenceSimple-Uncertainty": TwoPhaseAlgorithm(ConfidenceSimpleObjective(), UncertaintyObjective(use_cf=True), gp=build_gp()),
    }

ALGORITHMS = list(build_algorithms().keys())
