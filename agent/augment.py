import numpy as np

# Observation indices
# [p1.x,p1.y,p1.ang,p1.vx,p1.vy,p1.angvel,
#  p2.x,p2.y,p2.ang,p2.vx,p2.vy,p2.angvel,
#  puck.x,puck.y,puck.vx,puck.vy, t1,t2]
Y_POS_IDXS = [1, 7, 13]
Y_VEL_IDXS = [4, 10, 15]
ANGLE_IDXS = [2, 8]
ANGVEL_IDXS = [5, 11]

def flip_y_obs(obs: np.ndarray) -> np.ndarray:
    """Mirror top/bottom: y -> -y. Keeps goals the same, so reward is unchanged."""
    o = np.array(obs, dtype=np.float32, copy=True)
    o[Y_POS_IDXS] *= -1.0
    o[Y_VEL_IDXS] *= -1.0
    o[ANGLE_IDXS] *= -1.0
    o[ANGVEL_IDXS] *= -1.0
    return o

def flip_y_action(action: np.ndarray) -> np.ndarray:
    """Mirror action for y-flip. action = [fx, fy, torque, shoot]."""
    a = np.array(action, dtype=np.float32, copy=True)
    if a.shape[-1] >= 2:
        a[1] *= -1.0
    if a.shape[-1] >= 3:
        a[2] *= -1.0
    return a