import math
import contextlib
import sys

# world
GRID_H = 0.2 # [m]

# object
OBJ_D_MAX = 8 # [m]
OBJ_D_MIN = 2 # [m]
OBJ_D_AVG = lambda: (OBJ_D_MAX + OBJ_D_MIN) / 2

# camera
CAM_D = 10 # [m]
CAM_DOF = 10 # [m] 
CAM_FOV = math.radians(35) # [rad]
OBS_NOISE = 0.2 # standard deviation of observation noise


@contextlib.contextmanager
def local():
    # get reference to this module
    params = sys.modules[__name__]
    
    # save current parameters
    grid_h = params.GRID_H

    obj_d_max = params.OBJ_D_MAX
    obj_d_min = params.OBJ_D_MIN
    
    cam_d = params.CAM_D
    cam_dof = params.CAM_DOF
    cam_fov = params.CAM_FOV
    obs_noise = params.OBS_NOISE
    try:
        yield
    finally:
        # restore old parameters
        params.GRID_H = grid_h

        params.OBJ_D_MAX = obj_d_max
        params.OBJ_D_MIN = obj_d_min
        
        params.CAM_D = cam_d
        params.CAM_DOF = cam_dof
        params.CAM_FOV = cam_fov
        params.OBS_NOISE = obs_noise
