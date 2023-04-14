import math

# world
GRID_H = 0.2 # [m]

# object
OBJ_D_MAX = 8 # [m]
OBJ_D_MIN = 2 # [m]
OBJ_D_AVG = (OBJ_D_MAX + OBJ_D_MIN) / 2

# camera
CAM_D = 10 # [m]
CAM_FOV = 35 # [deg]
CAM_FOV_RAD = math.radians(CAM_FOV)
CAM_DOF = CAM_D * 1 # [m] 
OBS_NOISE = 0.2 # standard deviation of observation noise
