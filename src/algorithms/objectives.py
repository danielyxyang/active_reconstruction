import numpy as np

from parameters import OBJ_D_MAX, CAM_FOV_RAD, CAM_DOF, CAM_D, GRID_H
from simulation.camera import Camera
from utils.helpers import (
    cartesian_to_polar,
    cartesian_to_pixel,
    pixel_to_cartesian,
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
        if xlim is None:
            xlim = (-OBJ_D_MAX, OBJ_D_MAX)
        if ylim is None:
            ylim = (-OBJ_D_MAX, OBJ_D_MAX)
        # TODO GP can also be outside of OBJ_D_MAX!
        bbox_min = cartesian_to_pixel(xlim[0], ylim[0])
        bbox_max = cartesian_to_pixel(xlim[1], ylim[1])
        pixels = np.concatenate(np.stack(np.mgrid[
            bbox_min[0] : bbox_max[0] + 1,
            bbox_min[1] : bbox_max[1] + 1,
        ], axis=-1)).T
        pixel_centers = pixel_to_cartesian(*pixels)
        return pixel_centers

    @staticmethod
    def get_boundary_intersections(camera, gp):
        # compute intersections of FOV boundary and lower confidence bound
        lower, _ = gp.confidence_boundary()
        # duplicate lower boundary for wrap around
        if camera.theta < np.pi:
            lower = np.concatenate([  # add duplicate boundary to the left
                (lower[0] - 2*np.pi, lower[1]),
                (lower[0], lower[1]),
            ], axis=1)
        else:
            lower = np.concatenate([  # add duplicate boundary to the right
                (lower[0], lower[1]),
                (lower[0] + 2*np.pi, lower[1]),
            ], axis=1)
        # find intersection to the left (i.e. smaller polar angle)
        fov_boundary = camera.ray_f(CAM_FOV_RAD/2)(lower[0])
        fov_limit = camera.camera_to_polar(CAM_FOV_RAD/2, CAM_DOF)
        intersections = intersect_functions(fov_boundary, lower[1], mode="left")
        c1 = np.concatenate([lower[:, intersections].T, [fov_limit]]).T
        c1 = c1[:, is_in_range(c1[0], (fov_limit[0], camera.theta), mod=2*np.pi)]
        c1 = c1[:, np.argmin(camera.polar_to_camera(*c1)[1])]
        # find intersection to the right
        fov_boundary = camera.ray_f(-CAM_FOV_RAD/2)(lower[0])
        fov_limit = camera.camera_to_polar(-CAM_FOV_RAD/2, CAM_DOF)
        intersections = intersect_functions(fov_boundary, lower[1], mode="right")
        c2 = np.concatenate([lower[:, intersections].T, [fov_limit]]).T
        c2 = c2[:, is_in_range(c2[0], (camera.theta, fov_limit[0]), mod=2*np.pi)]
        c2 = c2[:, np.argmin(camera.polar_to_camera(*c2)[1])]
        return c1, c2


class SurfaceMarginalObjective(Objective):
    """Number of surface points on the object have not previously been observed."""

    def __init__(self, obj=None, **kwargs):
        super().__init__(**kwargs)
        self.obj = obj # must be set before objective is used
    
    def compute_estimate_points(self, camera, data):
        observations = data[Objective.OBSERVATIONS]
        
        # compute marginal observed surface points
        observed_points = camera.compute_observation(self.obj.surface_points)
        return setdiff2d(observed_points.T, observations.T).T


class IntersectionObjective(Objective):
    """Number of visible pixels within intersection of FOV and confidence region."""
    
    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # create list of candidate pixels in intersection
        # TODO improve (e.g. filter based on OBJ_D_MAX as circle, not square)
        pixel_centers = Objective.get_candidate_pixels()
        pixel_polar = cartesian_to_polar(*pixel_centers)
        # keep points in FOV
        pixel_polar = pixel_polar[:, camera.is_in_FOV(pixel_polar)]
        # (short-circuit) AND not occluded by lower confidence bound
        pixel_polar = pixel_polar[:, camera.is_not_occluded(pixel_polar, gp.lower_points)] # much faster to use discretized lower confidence bound
        # (short-circuit) AND in confidence region
        (_, lower), (_, upper) = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= 2*np.pi
        return pixel_polar


class SimpleIntersectionObjective(Objective):
    """Number of pixels within intersection of FOV and confidence region."""
    
    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # create list of candidate pixels in intersection
        # TODO improve (e.g. filter based on OBJ_D_MAX as circle, not square)
        pixel_centers = Objective.get_candidate_pixels()
        pixel_polar = cartesian_to_polar(*pixel_centers)
        # keep points in FOV
        pixel_polar = pixel_polar[:, camera.is_in_FOV(pixel_polar)]
        # (short-circuit) AND in confidence region
        (_, lower), (_, upper) = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= 2*np.pi
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        c1, c2 = self.get_boundary(camera, data)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = lower[0, 1] - lower[0, 0] # TODO improve
        lower = lower[:, is_in_range(lower[0], (c1[0], c2[0]), mod=2*np.pi)]
        upper = upper[:, is_in_range(upper[0], (c1[0], c2[0]), mod=2*np.pi)]
        # compute FOV boundary
        phi = lower[0]
        mask1 = is_in_range(phi, (c1[0], camera.theta), mod=2*np.pi)
        mask2 = is_in_range(phi, (camera.theta, c2[0]), mod=2*np.pi)
        fov = np.zeros_like(phi)
        fov[mask1] = camera.ray_f(CAM_FOV_RAD/2)(phi[mask1])
        fov[mask2] = camera.ray_f(-CAM_FOV_RAD/2)(phi[mask2])
        # compute number of pixels
        estimate = np.sum(np.maximum(np.minimum(upper[1], fov) ** 2 - lower[1] ** 2, 0) / 2 * delta_phi) / (GRID_H ** 2)
        return estimate
    
    def get_boundary(self, camera, data):
        c1 = camera.camera_to_polar(CAM_FOV_RAD/2, CAM_DOF)
        c2 = camera.camera_to_polar(-CAM_FOV_RAD/2, CAM_DOF)
        return c1, c2


class ConfidenceObjective(Objective):
    """Number of pixels within confidence region bounded by intersection points of FOV
    with confidence region."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # compute summation boundary
        c1, c2 = self.get_boundary(camera, data)
        # create list of candidate pixels in intersection
        pixel_centers = Objective.get_candidate_pixels()
        pixel_polar = cartesian_to_polar(*pixel_centers)
        # keep pixels within c1 and c2
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[0], (c1[0], c2[0]), mod=2*np.pi)]
        # keep pixels between upper and lower confidence bound
        (_, lower), (_, upper) = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= (2*np.pi)
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # compute summation boundary
        c1, c2 = self.get_boundary(camera, data)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = lower[0, 1] - lower[0, 0] # TODO improve
        lower = lower[:, is_in_range(lower[0], (c1[0], c2[0]), mod=2*np.pi)]
        upper = upper[:, is_in_range(upper[0], (c1[0], c2[0]), mod=2*np.pi)]
        # compute number of pixels
        estimate = np.sum((upper[1] ** 2 - lower[1] ** 2) / 2 * delta_phi) / (GRID_H ** 2)
        return estimate

    def get_boundary(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        return Objective.get_boundary_intersections(camera, gp)


class SimpleConfidenceObjective(Objective):
    """Number of pixels within confidence region bounded by endpoints of FOV."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        
        # compute summation boundary
        c1, c2 = self.get_boundary(camera, data)
        # create list of candidate pixels in intersection
        pixel_centers = Objective.get_candidate_pixels()
        pixel_polar = cartesian_to_polar(*pixel_centers)
        # keep pixels within c1 and c2
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[0], (c1[0], c2[0]), mod=2*np.pi)]
        # keep pixels between upper and lower confidence bound
        (_, lower), (_, upper) = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= (2*np.pi)
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        c1, c2 = self.get_boundary(camera, data)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = lower[0, 1] - lower[0, 0] # TODO improve
        lower = lower[:, is_in_range(lower[0], (c1[0], c2[0]), mod=2*np.pi)]
        upper = upper[:, is_in_range(upper[0], (c1[0], c2[0]), mod=2*np.pi)]
        # compute number of pixels
        estimate = np.sum((upper[1] ** 2 - lower[1] ** 2) / 2 * delta_phi) / (GRID_H ** 2)
        return estimate

    def get_boundary(self, camera, data):
        c1 = camera.camera_to_polar(CAM_FOV_RAD/2, CAM_DOF)
        c2 = camera.camera_to_polar(-CAM_FOV_RAD/2, CAM_DOF)
        return c1, c2


class PolarSimpleConfidenceObjective(Objective):
    """Number of polar pixels within confidence region bounded by endpoints of FOV."""

    def compute_estimate_points(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        c1, c2 = self.get_boundary(camera, data)
        # create list of candidate pixels in intersection
        pixel_centers = Objective.get_candidate_pixels(xlim=(0, 2*np.pi), ylim=(0, 10))
        pixel_polar = pixel_centers
        # keep pixels within c1 and c2
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[0], (c1[0], c2[0]), mod=2*np.pi)]
        # keep pixels between upper and lower confidence bound
        (_, lower), (_, upper) = gp.confidence_boundary(pixel_polar[0], interp=True)
        pixel_polar = pixel_polar[:, is_in_range(pixel_polar[1], (lower, upper))]
        pixel_polar[0] %= (2*np.pi)
        return pixel_polar

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        c1, c2 = self.get_boundary(camera, data)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = lower[0, 1] - lower[0, 0] # TODO improve
        lower = lower[:, is_in_range(lower[0], (c1[0], c2[0]), mod=2*np.pi)]
        upper = upper[:, is_in_range(upper[0], (c1[0], c2[0]), mod=2*np.pi)]
        # compute number of polar pixels
        estimate = np.sum((upper[1] - lower[1]) * delta_phi) / (GRID_H ** 2)
        return estimate

    def get_boundary(self, camera, data):
        c1 = camera.camera_to_polar(CAM_FOV_RAD/2, CAM_DOF)
        c2 = camera.camera_to_polar(-CAM_FOV_RAD/2, CAM_DOF)
        return c1, c2


class WeightedSimpleConfidenceObjective(Objective):
    """Number of pixels within confidence region bounded by endpoints of FOV and weighted by
    `FOV(phi) / CAM_D`."""

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]

        # compute summation boundary
        c1, c2 = self.get_boundary(camera, data)
        # compute upper and lower boundary
        lower, upper = gp.confidence_boundary()
        delta_phi = lower[0, 1] - lower[0, 0] # TODO improve
        lower = lower[:, is_in_range(lower[0], (c1[0], c2[0]), mod=2*np.pi)]
        upper = upper[:, is_in_range(upper[0], (c1[0], c2[0]), mod=2*np.pi)]
        # compute FOV boundary
        phi = lower[0]
        mask1 = is_in_range(phi, (c1[0], camera.theta), mod=2*np.pi)
        mask2 = is_in_range(phi, (camera.theta, c2[0]), mod=2*np.pi)
        fov = np.zeros_like(phi)
        fov[mask1] = camera.ray_f(CAM_FOV_RAD/2)(phi[mask1])
        fov[mask2] = camera.ray_f(-CAM_FOV_RAD/2)(phi[mask2])
        # compute weighted number of pixels
        estimate = np.sum((fov / CAM_D) * (upper[1] ** 2 - lower[1] ** 2) / 2 * delta_phi) / (GRID_H ** 2)
        return estimate
    
    def get_boundary(self, camera, data):
        c1 = camera.camera_to_polar(CAM_FOV_RAD/2, CAM_DOF)
        c2 = camera.camera_to_polar(-CAM_FOV_RAD/2, CAM_DOF)
        return c1, c2


# class Weighted2SimpleConfidenceObjective(Objective):
#     """Some docstring"""

#     def compute_estimate_cf(self, camera, data):
#         gp = data[Objective.CONFIDENCE]

#         # compute summation boundary
#         c1, c2 = self.get_boundary(camera, data)
#         # compute upper and lower boundary
#         lower, upper = gp.confidence_boundary()
#         delta_phi = lower[0, 1] - lower[0, 0] # TODO improve
#         lower = lower[:, is_in_range(lower[0], (c1[0], c2[0]), mod=2*np.pi)]
#         upper = upper[:, is_in_range(upper[0], (c1[0], c2[0]), mod=2*np.pi)]
#         # compute FOV boundary
#         phi = lower[0]
#         mask1 = is_in_range(phi, (c1[0], camera.theta), mod=2*np.pi)
#         mask2 = is_in_range(phi, (camera.theta, c2[0]), mod=2*np.pi)
#         fov = np.zeros_like(phi)
#         fov[mask1] = camera.ray_f(CAM_FOV_RAD/2)(phi[mask1])
#         fov[mask2] = camera.ray_f(-CAM_FOV_RAD/2)(phi[mask2])
#         # compute weighted number of pixels
#         estimate = np.sum((fov / CAM_D) * (upper[1] ** 2 - lower[1] ** 2) / 2 * delta_phi) / (GRID_H ** 2)
#         return estimate

#     def get_boundary(self, camera, data):
#         c1 = camera.camera_to_polar(CAM_FOV_RAD/2, CAM_DOF)
#         c2 = camera.camera_to_polar(-CAM_FOV_RAD/2, CAM_DOF)
#         return c1, c2


class UncertaintyObjective(Objective):
    """Area of circle sector between upper and lower confidence boundary with unit angle in radians."""

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        # compute upper and lower boundary at current camera position
        lower, upper = gp.confidence_boundary(camera.theta, interp=True)
        estimate = (upper[1] ** 2 - lower[1] ** 2) / 2
        return estimate


class PolarUncertaintyObjective(Objective):
    """Difference between upper and lower confidence boundary."""

    def compute_estimate_cf(self, camera, data):
        gp = data[Objective.CONFIDENCE]
        # compute upper and lower boundary at current camera position
        lower, upper = gp.confidence_boundary(camera.theta, interp=True)
        estimate = upper[1] - lower[1]
        return estimate
    

# class TemplateObjective(Objective):
#     """Some docstring"""

#     def compute_estimate_points(self, camera, data):
#         gp = data[Objective.CONFIDENCE]
        
#         return None
    
#     def compute_estimate_cf(self, camera, data):
#         gp = data[Objective.CONFIDENCE]

#         return None
    
#     def get_boundary(self, camera, data):
#         return None
