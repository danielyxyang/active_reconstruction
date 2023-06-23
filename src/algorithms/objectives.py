import numpy as np

import parameters as params
from simulation.camera import Camera
from utils.math import (
    cartesian_to_polar,
    cartesian_to_pixel,
    pixel_to_cartesian,
)
from utils_ext.math import (
    intersect_functions,
    is_in_range,
    setdiff2d,
)


class Objective():
    CONFIDENCE = "gp"
    OBSERVATIONS = "observations"

    def __init__(self, use_cf=False):
        self.closed_form = use_cf

    def __call__(self, thetas, data):
        """Compute estimated number of observed pixels for all camera locations."""
        return np.vectorize(lambda theta: self.compute_estimate(Camera(theta), data))(thetas)
    
    def compute_estimate(self, camera, data):
        """Compute estimated number of observed pixels for given camera."""
        if self.closed_form:
            return self.compute_estimate_cf(camera, data)
        else:
            return len(self.compute_estimate_points(camera, data).T)
    
    def compute_estimate_points(self, camera, data):
        """Compute list of estimated observed pixels for given camera."""
        print("WARNING: not implemented to evaluate objective function numerically")
        
    def compute_estimate_cf(self, camera, data):
        """Compute objective value with closed-form expression for given camera."""
        print("WARNING: not implemented to evaluate objective function analytically")

    # STATIC METHODS
    
    @staticmethod
    def get_candidate_pixels(xlim=None, ylim = None):
        """Return cartesian coordinates of candidate pixels for numerical objective functions."""
        # possible that GP is slightly outside of OBJ_D_MAX, but we assume it stays inside CAM_D
        if xlim is None:
            xlim = (-params.CAM_D, params.CAM_D)
        if ylim is None:
            ylim = (-params.CAM_D, params.CAM_D)
        bbox_min = cartesian_to_pixel(xlim[0], ylim[0])
        bbox_max = cartesian_to_pixel(xlim[1], ylim[1])
        pixels = np.concatenate(np.stack(np.mgrid[
            bbox_min[0] : bbox_max[0] + 1,
            bbox_min[1] : bbox_max[1] + 1,
        ], axis=-1)).T
        pixel_centers = pixel_to_cartesian(*pixels)
        return pixel_centers

    @staticmethod
    def get_FOV_confidence_intersection(camera, gp):
        """Return boundaries of FOV-confidence intersection summation interval."""
        # compute intersections of FOV boundary and lower confidence bound
        lower, _ = gp.confidence_boundary()
        # duplicate lower boundary for wrap around
        if camera.theta < np.pi:
            lower = np.concatenate([  # add duplicate boundary to the left
                (gp.x_eval - 2*np.pi, lower),
                (gp.x_eval, lower),
            ], axis=1)
        else:
            lower = np.concatenate([  # add duplicate boundary to the right
                (gp.x_eval, lower),
                (gp.x_eval + 2*np.pi, lower),
            ], axis=1)
        # find intersection to the left (i.e. smaller polar angle)
        fov_boundary = camera.ray_f(params.CAM_FOV_RAD()/2)(lower[0])
        fov_limit = camera.camera_to_polar(params.CAM_FOV_RAD()/2, params.CAM_DOF)
        intersections = intersect_functions(fov_boundary, lower[1], mode="left")
        phi1 = np.concatenate([lower[:, intersections].T, [fov_limit]]).T
        phi1 = phi1[:, is_in_range(phi1[0], (fov_limit[0], camera.theta), mod=2*np.pi)]
        phi1 = phi1[:, np.argmin(camera.polar_to_camera(*phi1)[1])]
        # find intersection to the right
        fov_boundary = camera.ray_f(-params.CAM_FOV_RAD()/2)(lower[0])
        fov_limit = camera.camera_to_polar(-params.CAM_FOV_RAD()/2, params.CAM_DOF)
        intersections = intersect_functions(fov_boundary, lower[1], mode="right")
        phi2 = np.concatenate([lower[:, intersections].T, [fov_limit]]).T
        phi2 = phi2[:, is_in_range(phi2[0], (camera.theta, fov_limit[0]), mod=2*np.pi)]
        phi2 = phi2[:, np.argmin(camera.polar_to_camera(*phi2)[1])]
        return phi1, phi2
    
    @staticmethod
    def get_simple_FOV_endpoint(camera):
        """Return boundaries of simple FOV endpoint summation interval"""
        phi1 = camera.camera_to_polar(params.CAM_FOV_RAD()/2, params.CAM_DOF)
        phi2 = camera.camera_to_polar(-params.CAM_FOV_RAD()/2, params.CAM_DOF)
        return phi1, phi2


class ObservedSurfaceMarginalObjective(Objective):
    """Number of newly observed surface points on the object."""

    def __init__(self, obj=None, **kwargs):
        super().__init__(**kwargs)
        self.obj = obj # must be set before objective is used
    
    def compute_estimate_points(self, camera, data):
        observations = data[Objective.OBSERVATIONS]
        
        # compute marginal observed surface points
        observed_points = camera.compute_observation(self.obj.surface_points)
        return setdiff2d(observed_points.T, observations.observed_points.T).T


class ObservedSurfaceObjective(Objective):
    """Number of observed surface points on the object."""

    def __init__(self, obj=None, **kwargs):
        super().__init__(**kwargs)
        self.obj = obj # must be set before objective is used
    
    def compute_estimate_points(self, camera, data):
        return camera.compute_observation(self.obj.surface_points)


class ObservedConfidenceLowerObjective(Objective):
    """Number of observed surface points on the lower confidence bound."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        return camera.compute_observation(gp.lower_points)


class ObservedConfidenceUpperObjective(Objective):
    """Number of observed surface points on the upper confidence bound."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        return camera.compute_observation(gp.upper_points)


class IntersectionOcclusionAwareObjective(Objective):
    """Number of visible pixels within intersection of FOV and confidence region."""
    
    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # create list of candidate pixels in intersection
        pixel_centers = Objective.get_candidate_pixels()
        pixel_polar = cartesian_to_polar(*pixel_centers)
        # keep points in FOV
        pixel_polar = pixel_polar[:, camera.is_in_FOV(pixel_polar)]
        # (short-circuit) AND not occluded by lower confidence bound
        pixel_polar = pixel_polar[:, camera.is_not_occluded(pixel_polar, gp.lower_points)] # much faster to use discretized lower confidence bound
        # (short-circuit) AND in confidence region
        lower, upper = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= 2*np.pi
        return pixel_polar

    def get_summation_interval(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        return Objective.get_FOV_confidence_intersection(camera, gp)


class IntersectionObjective(Objective):
    """Number of pixels within intersection of FOV and confidence region."""
    
    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # create list of candidate pixels in intersection
        pixel_centers = Objective.get_candidate_pixels()
        pixel_polar = cartesian_to_polar(*pixel_centers)
        # keep points in FOV
        pixel_polar = pixel_polar[:, camera.is_in_FOV(pixel_polar)]
        # (short-circuit) AND in confidence region
        lower, upper = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= 2*np.pi
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        mask = is_in_range(gp.x_eval, (phi1[0], phi2[0]), mod=2*np.pi)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = gp.x_eval[1] - gp.x_eval[0] # TODO improve
        lower = lower[mask]
        upper = upper[mask]
        # compute FOV boundary
        phi = gp.x_eval[mask]
        mask1 = is_in_range(phi, (phi1[0], camera.theta), mod=2*np.pi)
        mask2 = is_in_range(phi, (camera.theta, phi2[0]), mod=2*np.pi)
        fov = np.zeros_like(phi)
        fov[mask1] = camera.ray_f(params.CAM_FOV_RAD()/2)(phi[mask1])
        fov[mask2] = camera.ray_f(-params.CAM_FOV_RAD()/2)(phi[mask2])
        # compute number of pixels
        estimate = np.sum(1/2 * np.maximum(np.minimum(upper, fov)**2 - lower**2, 0) * delta_phi) / (params.GRID_H**2)
        return estimate
    
    def get_summation_interval(self, camera, data):
        return Objective.get_simple_FOV_endpoint(camera)


class ConfidenceObjective(Objective):
    """Number of pixels within confidence region bounded by intersection points of FOV with confidence region."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        # create list of candidate pixels in confidence region
        pixel_centers = Objective.get_candidate_pixels()
        pixel_polar = cartesian_to_polar(*pixel_centers)
        # keep pixels within phi1 and phi2
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[0], (phi1[0], phi2[0]), mod=2*np.pi)]
        # keep pixels between upper and lower confidence bound
        lower, upper = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= (2*np.pi)
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        mask = is_in_range(gp.x_eval, (phi1[0], phi2[0]), mod=2*np.pi)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = gp.x_eval[1] - gp.x_eval[0] # TODO improve
        lower = lower[mask]
        upper = upper[mask]
        # compute number of pixels
        estimate = np.sum(1/2 * (upper**2 - lower**2) * delta_phi) / (params.GRID_H**2)
        return estimate

    def get_summation_interval(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        return Objective.get_FOV_confidence_intersection(camera, gp)


class ConfidenceSimpleObjective(Objective):
    """Number of pixels within confidence region bounded by endpoints of FOV."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        # create list of candidate pixels in confidence region
        pixel_centers = Objective.get_candidate_pixels()
        pixel_polar = cartesian_to_polar(*pixel_centers)
        # keep pixels within phi1 and phi2
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[0], (phi1[0], phi2[0]), mod=2*np.pi)]
        # keep pixels between upper and lower confidence bound
        lower, upper = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= (2*np.pi)
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        mask = is_in_range(gp.x_eval, (phi1[0], phi2[0]), mod=2*np.pi)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = gp.x_eval[1] - gp.x_eval[0] # TODO improve
        lower = lower[mask]
        upper = upper[mask]
        # compute number of pixels
        estimate = np.sum(1/2 * (upper**2 - lower**2) * delta_phi) / (params.GRID_H**2)
        return estimate

    def get_summation_interval(self, camera, data):
        return Objective.get_simple_FOV_endpoint(camera)


class ConfidencePolarObjective(Objective):
    """Number of polar pixels within confidence region bounded by endpoints of FOV."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        # create list of candidate pixels in confidence region
        pixel_centers = Objective.get_candidate_pixels(xlim=(0, 2*np.pi), ylim=(0, 10))
        pixel_polar = pixel_centers
        # keep pixels within phi1 and phi2
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[0], (phi1[0], phi2[0]), mod=2*np.pi)]
        # keep pixels between upper and lower confidence bound
        lower, upper = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= (2*np.pi)
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        mask = is_in_range(gp.x_eval, (phi1[0], phi2[0]), mod=2*np.pi)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = gp.x_eval[1] - gp.x_eval[0] # TODO improve
        lower = lower[mask]
        upper = upper[mask]
        # compute number of polar pixels
        estimate = np.sum((upper - lower) * delta_phi) / (params.GRID_H**2)
        return estimate

    def get_summation_interval(self, camera, data):
        return Objective.get_simple_FOV_endpoint(camera)


class ConfidenceSimpleWeightedObjective(Objective):
    """Number of pixels within confidence region bounded by endpoints of FOV and weighted by `FOV(phi) / CAM_D`."""

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        mask = is_in_range(gp.x_eval, (phi1[0], phi2[0]), mod=2*np.pi)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = gp.x_eval[1] - gp.x_eval[0] # TODO improve
        lower = lower[mask]
        upper = upper[mask]
        # compute FOV boundary
        phi = gp.x_eval[mask]
        mask1 = is_in_range(phi, (phi1[0], camera.theta), mod=2*np.pi)
        mask2 = is_in_range(phi, (camera.theta, phi2[0]), mod=2*np.pi)
        fov = np.zeros_like(phi)
        fov[mask1] = camera.ray_f(params.CAM_FOV_RAD()/2)(phi[mask1])
        fov[mask2] = camera.ray_f(-params.CAM_FOV_RAD()/2)(phi[mask2])
        # compute weighted number of pixels
        estimate = np.sum(fov/params.CAM_D * 1/2 * (upper**2 - lower**2) * delta_phi) / (params.GRID_H**2)
        return estimate
    
    def get_summation_interval(self, camera, data):
        return Objective.get_simple_FOV_endpoint(camera)


class UncertaintyObjective(Objective):
    """Area of circle sector between upper and lower confidence boundary with unit angle in radians."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        # create list of candidate pixels in confidence region
        pixel_centers = Objective.get_candidate_pixels()
        pixel_polar = cartesian_to_polar(*pixel_centers)
        # keep pixels within phi1 and phi2
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[0], (phi1[0], phi2[0]), mod=2*np.pi)]
        # keep pixels within uncertainty at current camera location
        lower, upper = gp.confidence_boundary(camera.theta, interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= (2*np.pi)
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        # compute upper and lower boundary at current camera position
        lower, upper = gp.confidence_boundary(camera.theta, interp=True)
        estimate = 1/2 * (phi2[0] - phi1[0]) * (upper**2 - lower**2) / (params.GRID_H**2)
        return estimate
    
    def get_summation_interval(self, camera, data):
        return Objective.get_simple_FOV_endpoint(camera)


class UncertaintyPolarObjective(Objective):
    """Difference between upper and lower confidence boundary."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        # create list of candidate pixels in intersection
        pixel_centers = Objective.get_candidate_pixels(xlim=(0, 2*np.pi), ylim=(0, 10))
        pixel_polar = pixel_centers
        # keep pixels within phi1 and phi2
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[0], (phi1[0], phi2[0]), mod=2*np.pi)]
        # keep pixels within uncertainty at current camera location
        lower, upper = gp.confidence_boundary(camera.theta, interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= (2*np.pi)
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        # compute summation boundary
        phi1, phi2 = self.get_summation_interval(camera, data)
        # compute upper and lower boundary at current camera position
        lower, upper = gp.confidence_boundary(camera.theta, interp=True)
        estimate = (phi2[0] - phi1[0]) * (upper - lower) / (params.GRID_H**2)
        return estimate
    
    def get_summation_interval(self, camera, data):
        return Objective.get_simple_FOV_endpoint(camera)
    

# class TemplateObjective(Objective):
#     """Some docstring"""

#     def compute_estimate_points(self, camera, data):
#         gp = data[Objective.CONFIDENCE]
#         return None
    
#     def compute_estimate_cf(self, camera, data):
#         gp = data[Objective.CONFIDENCE]
#         return None
    
#     def get_summation_interval(self, camera, data):
#         return None
