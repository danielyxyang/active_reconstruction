import numpy as np

from algorithms.objectives import (
    Objective,
    ObservedSurfaceMarginalObjective,
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
        self.estimates = None

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.estimates = None
    
    def add_observation(self, observation, noise=0):
        super().add_observation(observation, noise=noise)
        self.estimates = None
    
    def compute_nbv(self):
        estimates = self.compute_estimates()
        nbv_theta = self.thetas[np.argmax(estimates)]
        return nbv_theta
    
    def compute_estimates(self):
        if self.estimates is None:
            self.estimates = self.objective(self.thetas, {
                Objective.CONFIDENCE: self.gp,
                Objective.OBSERVATIONS: self.observations,
            })
        return self.estimates


class TwoPhaseAlgorithm(Algorithm):
    """Class for greedy 2-phase algorithms.
    
    Phase 1: finding two boundaries maximizing objective 1
    Phase 2: finding NBV within boundaries maximizing objective 2."""

    def __init__(self, objective1, objective2, n=100, **kwargs):
        super().__init__(**kwargs)
        self.objective1 = objective1
        self.objective2 = objective2
        self.thetas = np.linspace(0, 2*np.pi, num=n)
        self.estimates1 = None
        self.estimates2 = None

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.estimates1 = None
        self.estimates2 = None
    
    def add_observation(self, observation, noise=0):
        super().add_observation(observation, noise=noise)
        self.estimates1 = None
        self.estimates2 = None
    
    def compute_nbv(self):
        estimates1, estimates2 = self.compute_estimates()
        # phase 1
        nbv_theta1 = self.thetas[np.argmax(estimates1)]
        nbv_c1, nbv_c2 = self.objective1.get_boundary(Camera(nbv_theta1), {
            Objective.CONFIDENCE: self.gp,
            Objective.OBSERVATIONS: self.observations,
        })
        mask = is_in_range(self.thetas, (nbv_c1[0], nbv_c2[0]), mod=2*np.pi)
        # phase 2
        nbv_theta2 = self.thetas[mask][np.argmax(estimates2[mask])]
        return nbv_theta2
    
    def compute_estimates(self):
        if self.estimates1 is None:
            self.estimates1 = self.objective1(self.thetas, {
                Objective.CONFIDENCE: self.gp,
                Objective.OBSERVATIONS: self.observations,
            })

        if self.estimates2 is None:
            self.estimates2 = self.objective2(self.thetas, {
                Objective.CONFIDENCE: self.gp,
                Objective.OBSERVATIONS: self.observations,
            })
        
        return self.estimates1, self.estimates2


def build_algorithms(build_gp=lambda: None, object=None):
    return {
        # ALGORITHM WITH TRUE OBJECTIVE
        "ObservedSurfaceMarginal": dict(
            algorithm=GreedyAlgorithm(ObservedSurfaceMarginalObjective(obj=object), gp=build_gp()),
            color="darkred",
        ),

        # ALGORITHMS WITH CONFIDENCE OBJECTIVES
        "IntersectionOcclusionAware": dict(
            algorithm=GreedyAlgorithm(IntersectionOcclusionAwareObjective(), gp=build_gp()),
            color="cyan",
        ),
        "Intersection": dict(
            algorithm=GreedyAlgorithm(IntersectionObjective(), gp=build_gp()),
            color="deepskyblue",
        ),
        # "IntersectionSimple_cf": dict(
        #     algorithm=GreedyAlgorithm(IntersectionObjective(use_cf=True), gp=build_gp()),
        #     color="steelblue",
        # ),
        "Confidence": dict(
            algorithm=GreedyAlgorithm(ConfidenceObjective(), gp=build_gp()),
            color="darkgoldenrod",
        ),
        # "Confidence_cf": dict(
        #     algorithm=GreedyAlgorithm(ConfidenceObjective(use_cf=True), gp=build_gp()),
        #     color="goldenrod",
        # ),
        "ConfidenceSimple": dict(
            algorithm=GreedyAlgorithm(ConfidenceSimpleObjective(), gp=build_gp()),
            color="darkorange",
        ),
        # "ConfidenceSimple_cf": dict(
        #     algorithm=GreedyAlgorithm(ConfidenceSimpleObjective(use_cf=True), gp=build_gp()),
        #     color="orange",
        # ),
        "ConfidenceSimplePolar": dict(
            algorithm=GreedyAlgorithm(ConfidencePolarObjective(), gp=build_gp()),
            color="limegreen",
        ),
        # "ConfidenceSimplePolar_cf": dict(
        #     algorithm=GreedyAlgorithm(ConfidencePolarObjective(use_cf=True), gp=build_gp()),
        #     color="lime",
        # ),
        "ConfidenceSimpleWeighted_cf": dict(
            algorithm=GreedyAlgorithm(ConfidenceSimpleWeightedObjective(use_cf=True), gp=build_gp()),
            color="magenta",
        ),

        # ALGORITHMS WITH UNCERTAINTY OBJECTIVES
        "Uncertainty": dict(
            algorithm=GreedyAlgorithm(UncertaintyObjective(), gp=build_gp()),
            color="steelblue",
        ),
        # "Uncertainty_cf": dict(
        #     algorithm=GreedyAlgorithm(UncertaintyObjective(use_cf=True), gp=build_gp()),
        #     color="steelblue",
        # ),
        "UncertaintyPolar": dict(
            algorithm=GreedyAlgorithm(UncertaintyPolarObjective(), gp=build_gp()),
            color="slategray",
        ),
        # "UncertaintyPolar_cf": dict(
        #     algorithm=GreedyAlgorithm(UncertaintyPolarObjective(use_cf=True), gp=build_gp()),
        #     color="slategray",
        # ),
        
        # MULTI-PHASE ALGORITHMS
        "TwoPhase-ConfidenceSimple-Uncertainty": dict(
            algorithm=TwoPhaseAlgorithm(ConfidenceSimpleObjective(), UncertaintyObjective(use_cf=True), gp=build_gp()),
            color="deeppink",
        ),
    }

ALGORITHMS = list(build_algorithms().keys())
ALGORITHM_COLORS = {name: algorithm["color"] for name, algorithm in build_algorithms().items()}
