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
        the line of sight of the camera.

        Args:
            phi (N array): Polar angles relative to world coordinate system.
            r (N array): Radial distances relative to world coordinate system.

        Returns:
            2xN array: Polar coordinates relative to camera coordinate system.
        """
        # world polar -> camera cartesian
        xc = params.CAM_D - r * np.cos(self.theta - phi)
        yc = r * np.sin(self.theta - phi)
        # camera cartesian -> camera polar
        alpha = np.arctan2(yc, xc)
        d = np.sqrt(yc ** 2 + xc ** 2)
        return np.array([alpha, d])
    
    def camera_to_polar(self, alpha, d):
        """Convert camera polar coordinates to world polar coordinates.
        
        The zero polar angle in the camera coordinate system is aligned with
        the line of sight of the camera.

        Args:
            alpha (N array): Polar angles relative to camera coordinate system.
            d (N array): Radial distances relative to camera coordinate system.

        Returns:
            2xN array: Polar cooridnates relative to world coordinate system.
        """        
        # camera polar -> "world" cartesian (rotated by theta)
        xc = params.CAM_D - d * np.cos(-alpha)
        yc = d * np.sin(-alpha)
        # "world" cartesian (rotated by theta) -> world polar
        phi = np.arctan2(yc, xc) + self.theta
        r = np.sqrt(yc ** 2 + xc ** 2)
        return np.array([phi, r])

    def compute_observation(self, points, check_dof=True):
        """Compute list of observed points among the given points.
        
        Args:
            points (2xN array): Polar coordinates of points.
            check_dof (bool): Flag whether to take depth of field into
                consideration or not.
        
        Returns:
            2xM array: Polar coordinates of observed points.
        """
        alpha_min = 0
        alpha_max = 0
        def is_visible(alpha):
            # compare angle to previously computed angles
            nonlocal alpha_min, alpha_max
            is_occluded = alpha_min <= alpha <= alpha_max
            alpha_min = min(alpha_min, alpha)
            alpha_max = max(alpha_max, alpha)
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
        alpha, _ = self.polar_to_camera(*points[i])
        CHECKER = LoopChecker("Camera:compute_observation.1")
        while self.is_in_FOV(points[i], check_dof=check_dof) and alpha <= 0 and i != center_point_index:
            CHECKER()
            if is_visible(alpha):
                observation.append(points[i])
            i = (i + 1) % len(points)
            alpha, _ = self.polar_to_camera(*points[i])

        # iterate over surface points with decreasing world polar angle (i.e. to the left of LOS)
        i = (center_point_index - 1) % len(points)
        alpha, _ = self.polar_to_camera(*points[i])
        CHECKER = LoopChecker("Camera:compute_observation.2")
        while self.is_in_FOV(points[i], check_dof=check_dof) and alpha >= 0 and i != center_point_index:
            CHECKER()
            if is_visible(alpha):
                observation.append(points[i])
            i = (i - 1) % len(points)
            alpha, _ = self.polar_to_camera(*points[i])
        
        if len(observation) > 0:
            return np.array(observation).T
        else:
            return np.array([[], []])
    
    def is_in_FOV(self, points, check_dof=True):
        """Check whether the points lie in the FOV of camera.
        
        Args:
            points (2xN array): Polar coordinates of points.
            check_dof (bool): Flag whether to take depth of field into
                consideration or not.
        
        Returns:
            N array: Boolean flags for each point.
        """
        alpha, d = self.polar_to_camera(*points)
        return np.logical_and(np.abs(alpha) <= params.CAM_FOV_RAD()/2, d <= params.CAM_DOF if check_dof else True)
    
    def is_not_occluded(self, points, object):
        """Check whether the points are not occluded by the object points.
        
        Args:
            points (2xN array): Polar coordinates of points.
            object (2xM array): Polar coordinates of points on object.
        
        Returns:
            N array: Boolean flags for each point.
        """
        object_observed = self.compute_observation(object, check_dof=False)
        if len(object_observed[0]) == 0:
            return np.full(len(points[0]), True)

        points_alpha, points_d = self.polar_to_camera(*points)
        # keep points definitely not occluded by object
        obj_alpha, _ = self.polar_to_camera(*object)
        mask = np.logical_or(points_alpha < np.min(obj_alpha), np.max(obj_alpha) < points_alpha)
        mask_neg = np.logical_not(mask)
        # (short-circuit) OR with distance closer to camera then linearly interpolated observed points on object
        obj_alpha, obj_d = self.polar_to_camera(*object_observed)
        sorted = np.argsort(obj_alpha)
        mask[mask_neg] = points_d[mask_neg] < np.interp(points_alpha[mask_neg], obj_alpha[sorted], obj_d[sorted])

        return mask
    
    def ray_f(self, alpha):
        """Return polar function describing casted ray in polar coordinates.

        Args:
            alpha (scalar): Angle of ray relative to camera LOS.
        
        Returns:
            func: Polar function parameterizing casted ray.
        """        
        x, y = polar_to_cartesian(self.theta, params.CAM_D)
        line_slope = self.theta + alpha
        # check for numerical stability of tan and cot
        if (line_slope + np.pi/4) % np.pi < np.pi / 2: # use y = mx + b
            m = np.sin(line_slope) / np.cos(line_slope) # tan
            b = y - m*x
            return lambda phi: b / (np.sin(phi) - m * np.cos(phi))
        else: # use x = my + b
            m = np.cos(line_slope) / np.sin(line_slope) # cot (does not exist in numpy)
            b = x - m*y
            return lambda phi: b / (np.cos(phi) - m * np.sin(phi))
