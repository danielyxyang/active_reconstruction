import numpy as np

import parameters as params
from utils.math import polar_to_cartesian
from utils.tools import LoopChecker


class Camera():
    def __init__(self, theta=0):
        self.theta = theta
        self.observation = np.array([])
    
    def move(self, theta):
        self.theta = theta
    
    def observe(self, points):
        self.observation = self.compute_observation(points)
    
    def polar_to_camera(self, phi, r):
        """Convert world polar coordinates to camera polar coordinates.
        
        The zero polar angle in the camera coordinate system is aligned with
        the line of sight of the camera."""
        A = np.sin(phi - self.theta) * r
        B = np.cos(phi - self.theta) * r
        beta = -np.arctan2(A, params.CAM_D - B)
        d = np.sqrt(A ** 2 + (params.CAM_D - B) ** 2)
        # TODO draft
        # los = self.theta + np.pi
        # x, y = (polar_to_cartesian(phi, r).T - polar_to_cartesian(self.theta, params.CAM_D)).T
        # beta, d = cartesian_to_polar(x, y) - np.array([los, 0])
        return np.array([beta, d])
    
    def camera_to_polar(self, beta, d):
        """Convert camera polar coordinates to world polar coordinates.
        
        The zero polar angle in the camera coordinate system is aligned with
        the line of sight of the camera."""
        A = np.sin(-beta) * d
        B = params.CAM_D - np.cos(-beta) * d
        phi = np.arctan2(A, B) + self.theta
        r = np.sqrt(A ** 2 + B ** 2)
        # TODO draft
        # los = self.theta + np.pi
        # x, y = (polar_to_cartesian(beta + los, d).T + polar_to_cartesian(self.theta, params.CAM_D)).T
        # phi, r = cartesian_to_polar(x, y)
        return np.array([phi, r])

    def compute_observation(self, points, check_dof=True):
        """Compute list of observed points among the given points.
        
        Args:
            points: 2xN array containing polar coordinates of points
            check_dof: flag whether to take depth of field into consideration or not
        
        Returns:
            list: 2xM array containing polar coordinates of observed points
        """
        beta_min = 0
        beta_max = 0
        def is_visible(beta):
            # compare angle to previously computed angles
            nonlocal beta_min, beta_max
            is_occluded = beta_min <= beta <= beta_max
            beta_min = min(beta_min, beta)
            beta_max = max(beta_max, beta)
            return not is_occluded
        
        # initialize
        observation = []
        points = points.T
        
        # determine surface point at line of sight of camera
        center_point_index = np.argmin(np.abs(points[:, 0] - self.theta))
        if self.is_in_FOV(points[center_point_index], check_dof=check_dof):
            observation.append(points[center_point_index])
        
        # iterate over surface points with increasing world polar angle (i.e. to the right of LOS)
        i = (center_point_index + 1) % len(points)
        beta, _ = self.polar_to_camera(*points[i])
        CHECKER = LoopChecker("Camera:compute_observation.1")
        while self.is_in_FOV(points[i], check_dof=check_dof) and beta <= 0 and i != center_point_index:
            CHECKER()
            if is_visible(beta):
                observation.append(points[i])
            i = (i + 1) % len(points)
            beta, _ = self.polar_to_camera(*points[i])

        # iterate over surface points with decreasing world polar angle (i.e. to the left of LOS)
        i = (center_point_index - 1) % len(points)
        beta, _ = self.polar_to_camera(*points[i])
        CHECKER = LoopChecker("Camera:compute_observation.2")
        while self.is_in_FOV(points[i], check_dof=check_dof) and beta >= 0 and i != center_point_index:
            CHECKER()
            if is_visible(beta):
                observation.append(points[i])
            i = (i - 1) % len(points)
            beta, _ = self.polar_to_camera(*points[i])
        
        if len(observation) > 0:
            return np.array(observation).T
        else:
            return np.array([[], []])
    
    def is_in_FOV(self, points, check_dof=True):
        """Check whether the points lie in the FOV of camera.
        
        Args:
            points: 2xN array containing world polar coordinates of points
            check_dof: flag whether to take depth of field into consideration or not
        
        Returns
            list: N-dimensional boolean array
        """
        beta, d = self.polar_to_camera(*points)
        return np.logical_and(np.abs(beta) <= params.CAM_FOV_RAD/2, d <= params.CAM_DOF if check_dof else True)
    
    def is_not_occluded(self, points, object):
        """Check whether the points are not occluded by the object points.
        
        Args:
            points: 2xN array containing world polar coordinates of points
            object: 2xM array containing world polar coordinates of points on object
        
        Returns
            list: N-dimensional boolean array
        """
        object_observed = self.compute_observation(object, check_dof=False)
        if len(object_observed[0]) == 0:
            return np.full(len(points[0]), True)

        points_beta, points_d = self.polar_to_camera(*points)
        # keep points definitely not occluded by object
        obj_beta, _ = self.polar_to_camera(*object)
        mask = np.logical_or(points_beta < np.min(obj_beta), np.max(obj_beta) < points_beta)
        mask_neg = np.logical_not(mask)
        # (short-circuit) OR with distance closer to camera then linearly interpolated observed points on object
        obj_beta, obj_d = self.polar_to_camera(*object_observed)
        sorted = np.argsort(obj_beta)
        mask[mask_neg] = points_d[mask_neg] < np.interp(points_beta[mask_neg], obj_beta[sorted], obj_d[sorted])

        return mask
    
    def ray_f(self, beta):
        """Return function describing casted ray in polar coordinates."""
        x, y = polar_to_cartesian(self.theta, params.CAM_D)
        alpha = self.theta + beta
        # check for numerical stability of tan and cot
        if (alpha + np.pi/4) % np.pi < np.pi / 2: # use y = mx + b
            m = np.sin(alpha) / np.cos(alpha) # tan
            b = y - m*x
            return lambda phi: b / (np.sin(phi) - m * np.cos(phi))
        else: # use x = my + b
            m = np.cos(alpha) / np.sin(alpha) # cot (does not exist in numpy)
            b = x - m*y
            return lambda phi: b / (np.cos(phi) - m * np.sin(phi))
