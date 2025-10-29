
import numpy as np

# Assuming Lx, Ly, and max_vel are defined globally or passed as arguments
# For this script, we'll keep them as in the original cell, assuming they are defined elsewhere.
# Let's define them here for the script to be runnable standalone, adjust as needed.
Lx, Ly = 1.0, 1.0 # Default values, replace with actual if needed
max_vel = [np.pi, np.pi/2.1] # Default values, replace with actual if needed


def clip_state(x):
    # position inside Q, velocity limits
    x = x.copy()
    x[0] = np.clip(x[0], 0, Lx)
    x[1] = np.clip(x[1], 0, Ly)
    x[2] = np.clip(x[2], -max_vel[0], max_vel[0])
    x[3] = np.clip(x[3], -max_vel[1], max_vel[1])
    return x
