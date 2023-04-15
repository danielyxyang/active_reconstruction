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
from simulation.camera import Camera
from utils.math import is_in_range

TRUE_ALGORITHM = "Greedy-ObservedSurfaceMarginal"


class Algorithm():
    """Base class for algorithms."""
    
    def __init__(self, gp=None):
        self.observations = np.array([[], []])
        self.gp = gp

    def reset(self, algorithm=None):
        if algorithm is None:
            self.observations = np.array([[], []])
            self.gp.reset()
        else:
            self.observations = algorithm.observations
            self.gp = algorithm.gp

    def add_observation(self, observation, noise=0):
        self.observations = np.concatenate([self.observations, observation], axis=-1)
        self.gp.update(*observation, noise=noise)

    def compute_nbv(self):
        return 0


class GreedyAlgorithm(Algorithm):
    """Class for greedy algorithms maximizing some objective function."""
    
    def __init__(self, objective, n=100, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.thetas = np.linspace(0, 2*np.pi, num=n)

    def compute_nbv(self, with_estimates=False):
        # find NBV
        estimates = self.objective(self.thetas, {
            Objective.CONFIDENCE: self.gp,
            Objective.OBSERVATIONS: self.observations,
        })
        nbv = self.thetas[np.argmax(estimates)]
        # return NBV and optionally estimates
        if not with_estimates:
            return nbv
        else:
            return nbv, estimates
    
    def compute_estimate_points(self, camera):
        return self.objective.compute_estimate_points(camera, {
            Objective.CONFIDENCE: self.gp,
            Objective.OBSERVATIONS: self.observations,
        })


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
        estimates1 = self.objective1(self.thetas, {
            Objective.CONFIDENCE: self.gp,
            Objective.OBSERVATIONS: self.observations,
        })
        nbv1 = self.thetas[np.argmax(estimates1)]
        nbv1_phi1, nbv1_phi2 = self.objective1.get_summation_interval(Camera(nbv1), {
            Objective.CONFIDENCE: self.gp,
            Objective.OBSERVATIONS: self.observations,
        })
        mask = is_in_range(self.thetas, (nbv1_phi1[0], nbv1_phi2[0]), mod=2*np.pi)
        # phase 2
        estimates2 = self.objective2(self.thetas, {
            Objective.CONFIDENCE: self.gp,
            Objective.OBSERVATIONS: self.observations,
        })
        estimates2[np.logical_not(mask)] = np.nan
        nbv2 = self.thetas[np.nanargmax(estimates2)]
        # return NBV and optionally estimates
        if not with_estimates:
            return nbv2
        else:
            return nbv2, np.array([estimates1, estimates2])
    

def build_algorithms(build_gp=lambda: None, object=None):
    return {
        # greedy algorithms + observation-based objective functions
        TRUE_ALGORITHM: dict(
            algorithm=GreedyAlgorithm(ObservedSurfaceMarginalObjective(obj=object), gp=build_gp()),
            color="darkred",
        ),
        "Greedy-ObservedSurface": dict(
            algorithm=GreedyAlgorithm(ObservedSurfaceObjective(obj=object), gp=build_gp()),
            color="red",
        ),
        "Greedy-ObservedConfidenceLower": dict(
            algorithm=GreedyAlgorithm(ObservedConfidenceLowerObjective(), gp=build_gp()),
            color="dimgray",
        ),
        "Greedy-ObservedConfidenceUpper": dict(
            algorithm=GreedyAlgorithm(ObservedConfidenceUpperObjective(), gp=build_gp()),
            color="darkgray",
        ),

        # greedy algorithms + intersection-based objective functions
        "Greedy-IntersectionOcclusionAware": dict(
            algorithm=GreedyAlgorithm(IntersectionOcclusionAwareObjective(), gp=build_gp()),
            color="lime",
        ),
        "Greedy-Intersection": dict(
            algorithm=GreedyAlgorithm(IntersectionObjective(), gp=build_gp()),
            color="limegreen",
        ),
        "Greedy-Intersection_cf": dict(
            algorithm=GreedyAlgorithm(IntersectionObjective(use_cf=True), gp=build_gp()),
            color="limegreen",
        ),

        # greedy algorithms + confidence-based objective functions
        "Greedy-Confidence": dict(
            algorithm=GreedyAlgorithm(ConfidenceObjective(), gp=build_gp()),
            color="orange",
        ),
        "Greedy-Confidence_cf": dict(
            algorithm=GreedyAlgorithm(ConfidenceObjective(use_cf=True), gp=build_gp()),
            color="orange",
        ),
        "Greedy-ConfidenceSimple": dict(
            algorithm=GreedyAlgorithm(ConfidenceSimpleObjective(), gp=build_gp()),
            color="darkorange",
        ),
        "Greedy-ConfidenceSimple_cf": dict(
            algorithm=GreedyAlgorithm(ConfidenceSimpleObjective(use_cf=True), gp=build_gp()),
            color="darkorange",
        ),
        "Greedy-ConfidenceSimplePolar": dict(
            algorithm=GreedyAlgorithm(ConfidencePolarObjective(), gp=build_gp()),
            color="gold",
        ),
        "Greedy-ConfidenceSimplePolar_cf": dict(
            algorithm=GreedyAlgorithm(ConfidencePolarObjective(use_cf=True), gp=build_gp()),
            color="gold",
        ),
        "Greedy-ConfidenceSimpleWeighted_cf": dict(
            algorithm=GreedyAlgorithm(ConfidenceSimpleWeightedObjective(use_cf=True), gp=build_gp()),
            color="goldenrod",
        ),

        # greedy algorithms + uncertainty-based objective functions
        "Greedy-Uncertainty": dict(
            algorithm=GreedyAlgorithm(UncertaintyObjective(), gp=build_gp()),
            color="steelblue",
        ),
        "Greedy-Uncertainty_cf": dict(
            algorithm=GreedyAlgorithm(UncertaintyObjective(use_cf=True), gp=build_gp()),
            color="steelblue",
        ),
        "Greedy-UncertaintyPolar": dict(
            algorithm=GreedyAlgorithm(UncertaintyPolarObjective(), gp=build_gp()),
            color="deepskyblue",
        ),
        "Greedy-UncertaintyPolar_cf": dict(
            algorithm=GreedyAlgorithm(UncertaintyPolarObjective(use_cf=True), gp=build_gp()),
            color="deepskyblue",
        ),
        
        # two-phase algorithms
        "TwoPhase-ConfidenceSimple-Uncertainty": dict(
            algorithm=TwoPhaseAlgorithm(ConfidenceSimpleObjective(), UncertaintyObjective(use_cf=True), gp=build_gp()),
            color="magenta",
        ),
    }

ALGORITHMS = list(build_algorithms().keys())
ALGORITHM_COLORS = {name: algorithm["color"] for name, algorithm in build_algorithms().items()}
