
import casadi as ca

# Assuming dt and L are defined globally or passed as arguments
# For this script, we'll keep them as in the original cell, assuming they are defined elsewhere.
# Let's define them here for the script to be runnable standalone, adjust as needed.
dt = 0.2 # Default value, replace with actual if needed
L = 0.005 # Default value, replace with actual if needed

def dynamic_model(x, u):
  # x = [px, py, theta, gamma] ; theta = yaw, gamma = steering angle
  # u = [v, delta] ; v = veloctiy, delta = steering rate
  px = x[0]
  py = x[1]
  theta = x[2]
  gamma = x[3]
  v = u[0]
  delta = u[1]
  px_next = px + v * ca.cos(theta) * dt
  py_next = py + v * ca.sin(theta) * dt
  theta_next = theta + v / L * ca.tan(gamma) * dt
  gamma_next = gamma + delta * dt
  return ca.vertcat(px_next, py_next, theta_next, gamma_next)
