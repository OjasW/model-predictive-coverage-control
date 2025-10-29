
import numpy as np
import casadi as ca
import time
import matplotlib.pyplot as plt
from shapely.geometry import box, Point, Polygon
from scipy.linalg import solve_discrete_are # Import solve_discrete_are for dlqr

# Import the functions from the mpcc directory
from mpcc.bounded_voronoi import bounded_voronoi
from mpcc.weighted_centroid_of_polygon import weighted_centroid_of_polygon
from mpcc.phi_uniform import phi_uniform
from mpcc.dynamic_model import dynamic_model
from mpcc.clip_state import clip_state
from mpcc.coverage_cost import approx_coverage_cost # Import coverage_cost

# -------------------------
# Environment setup from initialisation cell
# -------------------------
Lx, Ly = 1.0, 1.0           # workspace size
Q_region = box(0, 0, Lx, Ly) # Define Q_region here since it's used in this file

M = 4                         # number of robots
dt = 0.2                      # timestep (s)
sim_steps = 100               # max simulation iterations
grid_density = 30             # grid per cell side for centroid integration (higher -> more accurate)
nx = 4                        # state size
nu = 2                        # input size
N_mpc = 10                    # MPC horizon
max_vel = [np.pi, np.pi/2.1]
max_acc = np.array([1.0, 1.0])
L = 0.005 # Vehicle Wheelbase

# -------------------------
# Cost weights
# -------------------------
Q_stage = np.diag([1.9, 1.9, 0.02, 0.01])   # state error weight (from the Algo 1 & 2 section)
R_stage = np.diag([0.08, 0.02])               # input weight (from the Algo 1 & 2 section)
# Q_terminal is computed via LQR

# -------------------------
# Linearizer and DLQR
# -------------------------
def linearize_dynamics(xbar, ubar, nx=nx, nu=nu):
    
    #Compute linearization A,B of f around (xbar, ubar).
    #Returns numeric numpy arrays A, B.
    
    X = ca.SX.sym('X', nx)
    U = ca.SX.sym('U', nu)

    # Create a CasADi function for the dynamics
    f_sym = dynamic_model(X, U)
    f_func = ca.Function('f_func', [X, U], [f_sym])

    # Compute Jacobians symbolically
    A_sym = ca.jacobian(f_sym, X)
    B_sym = ca.jacobian(f_sym, U)

    # Create a CasADi function for the Jacobians
    jac_func = ca.Function('jac_func', [X, U], [A_sym, B_sym])

    # Evaluate Jacobians numerically at the linearization point
    A_c, B_c = jac_func(xbar, ubar)
    A = np.array(A_c)
    B = np.array(B_c)
    return A, B

def dlqr(A, B, Q, R):
    #Discrete LQR via solve_discrete_are. Returns K,P (numpy).
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    # P = 0.5 * (P + P.T)  # ensure symmetry -- why is this needed?
    return K, P

# -------------------------
# MPC Builder With Terminal LQR
# -------------------------
def build_mpc_problem(nx=nx, nu=nu, N=N_mpc, Q_stage=Q_stage, R_stage=R_stage,
                      u_min=-max_acc, u_max=max_acc):
    # Building the linearizer problem
    # Reference is position independent, and terminal velocity & acceleration is 0
    # Linearize around a state with small non-zero velocity for controllability
    xbar_lin = np.zeros(nx)
    ubar_lin = np.zeros(nu)
    xbar_lin[2] = 0.1  # Small non-zero velocity in x direction
    ubar_lin[0] = 0.1  # Small non-zero velocity
    # A, B
    A_lin, B_lin = linearize_dynamics(xbar_lin, ubar_lin, nx=nx, nu=nu)

    # K, P
    P, K = dlqr(A_lin, B_lin, Q_stage, R_stage)

    # alpha for terminal constraint: x_N' P x_N <= alpha
    # Calculate alpha based on input constraints u = -Kx_N
    # (-max_acc <= -Kx_N <= max_acc) => (-max_acc <= -K(X_N - xbar) <= max_acc) assuming xbar=0
    # Based on the paper (eq 20), alpha is calculated from the maximum value of x^T P x
    # for x such that ||Kx||_inf <= max_acc. Here, we approximate with a conservative margin.
    # A more rigorous method involves finding the maximum eigenvalue of K^T R K * P^-1
    # or solving an optimization problem.
    # For simplicity, we use a common heuristic related to the LQR cost function.
    # alpha = a_max^T R a_max / (lambda_max(K^T R K))
    # Let's use a simpler heuristic related to V = x^T P x
    # Terminal cost V(x_N - xbar) <= alpha
    # For linearization point xbar_lin = [0,0,0.1,0], ubar_lin = [0.1,0]
    # We can choose alpha based on some desired level set of the terminal cost.
    # A common approach is to choose alpha such that the LQR controller's inputs
    # don't violate constraints near the origin.
    # Let's assume the terminal state is close to xbar. u_N = -K(x_N - xbar)
    # We need ||u_N||_inf <= max_acc
    # ||-K(x_N - xbar)||_inf <= max_acc
    # max_j |(K(x_N - xbar))_j| <= max_acc_j
    # Consider the quadratic form x_diff^T P x_diff <= alpha
    # We can find the max alpha such that there exists a state x_N where x_diff = x_N - xbar
    # satisfies ||-K x_diff||_inf <= max_acc AND x_diff^T P x_diff = alpha
    # This is related to the control invariant set.
    # A conservative alpha can be derived from the maximum allowable state deviation under LQR.
    # Let's use a simple heuristic: Choose alpha such that a state deviation x_diff
    # resulting in max acceleration command satisfies x_diff^T P x_diff = alpha.
    # If ||K x_diff||_inf = max_acc, what is x_diff^T P x_diff?
    # Let's use a simpler approach related to the cost function value.
    # The terminal cost is V(x_N) = (x_N - xbar)^T P (x_N - xbar). We want V(x_N) <= alpha.
    # Consider the state deviation x_diff that results in the maximum allowed input under LQR:
    # max_j |(K x_diff)_j| = max_acc_j
    # This boundary defines an ellipsoid in state space. We want alpha to be the maximum
    # value of x_diff^T P x_diff on this boundary.
    # The largest ellipsoid x_diff^T P x_diff <= alpha contained in ||K x_diff||_inf <= max_acc
    # has alpha = 1 / max_j ( (K^T (R)^-1 K)_j,j / P_j,j ) -- this is for continuous time
    # For discrete time, it's more complex. A common approach: find alpha such that
    # the LQR control for any state on the boundary x_diff^T P x_diff = alpha
    # satisfies the input constraints.
    # A simpler heuristic: Pick a state x_boundary such that ||K x_boundary||_inf = max_acc
    # and set alpha = x_boundary^T P x_boundary. This is still complex.

    # Let's use a more direct approach from common MPC practices for terminal sets.
    # Find the largest alpha such that for all x_N with (x_N - xbar)^T P (x_N - xbar) <= alpha,
    # the LQR control u_N = -K(x_N - xbar) satisfies ||u_N||_inf <= max_acc.
    # This means for all x_diff such that x_diff^T P x_diff <= alpha, we have |(K x_diff)_j| <= max_acc_j for j=0,1.
    # This is equivalent to: for each j, max_{x_diff: x_diff^T P x_diff <= alpha} |(K x_diff)_j| <= max_acc_j.
    # The maximum of |(K x_diff)_j| over x_diff^T P x_diff <= alpha is sqrt(alpha * (K P^-1 K^T)_j,j).
    # So we need sqrt(alpha * (K P^-1 K^T)_j,j) <= max_acc_j for all j.
    # alpha * (K P^-1 K^T)_j,j <= max_acc_j^2
    # alpha <= max_acc_j^2 / (K P^-1 K^T)_j,j
    # We need this to hold for all j, so alpha <= min_j { max_acc_j^2 / (K P^-1 K^T)_j,j }
    # (K P^-1 K^T) is nu x nu matrix.
    K_invP_KT = K @ np.linalg.solve(P, K.T) # More numerically stable than K @ np.linalg.inv(P) @ K.T
    alpha_candidates = []
    for j in range(nu):
        # Check if the diagonal element is positive to avoid division by zero or negative
        if K_invP_KT[j, j] > 1e-9: # Use a small tolerance
             alpha_candidates.append((max_acc[j]**2) / K_invP_KT[j, j])
        else:
            # If K P^-1 K^T is zero or negative on the diagonal, this direction might not be controllable or max_acc is effectively infinite
            # or there's an issue with the LQR design/linearization.
            # For robustness, we can set a very large alpha or handle this case specifically.
            # Assuming a well-posed problem, this shouldn't happen with appropriate Q_stage, R_stage.
            alpha_candidates.append(np.inf) # Effectively no constraint from this input

    # If alpha_candidates is empty (e.g., nu=0) or all are inf, choose a default small alpha
    if not alpha_candidates or all(np.isinf(a) for a in alpha_candidates):
        alpha = 1e-3 # Default small value
    else:
        alpha = min(alpha_candidates)
        # Add a safety margin, e.g., 90% of the maximum feasible alpha
        alpha *= 0.9

    # Build MPC
    X = ca.SX.sym('X', nx, N+1)
    U = ca.SX.sym('U', nu, N)
    x0 = ca.SX.sym('x0', nx)
    xbar = ca.SX.sym('xbar', nx)  # steady state to track
    ubar = ca.SX.sym('ubar', nu)
    # stage cost matrices as CasADi
    Qc = ca.DM(Q_stage)
    Rc = ca.DM(R_stage)
    QN = ca.DM(P) # Use P for terminal cost
    obj = 0
    g = []
    # initial cond
    g.append(X[:,0] - x0)
    for k in range(N):
        # dynamics: explicit Euler discrete mapping f_dynamics
        x_next = dynamic_model(X[:,k], U[:,k])
        g.append(X[:,k+1] - x_next)
        dx = X[:,k] - xbar
        du = U[:,k] - ubar
        obj = obj + ca.mtimes([dx.T, Qc, dx]) + ca.mtimes([du.T, Rc, du])
    # terminal cost
    dxN = X[:,N] - xbar
    obj = obj + ca.mtimes([dxN.T, QN, dxN])
    # Adding terminal constraint (x_N - xbar)' P (x_N - xbar) - α <= 0
    # Need to use (X[:, N] - xbar) as the state deviation for the terminal constraint
    g.append(ca.mtimes([(X[:, N] - xbar).T, QN, (X[:, N] - xbar)]) - float(alpha))
    # Pack decision variables
    dec_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    # Pack NLP
    g_all = ca.vertcat(*g)
    p = ca.vertcat(x0, xbar, ubar)
    nlp = {'x': dec_vars, 'f': obj, 'g': g_all, 'p': p}
    # Create solver
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    return solver, {'P': P, 'K': K, 'alpha': alpha}

# -------------------------
# MPC Solver With Terminal LQR
# -------------------------
def solve_mpc_casadi(solver, P, alpha, x0_val, xbar_val, ubar_val, nx=nx, nu=nu, N=N_mpc):
    # initial guess
    x_init = np.tile(x0_val.reshape(-1,1), (1, N+1))
    u_init = np.zeros((nu, N))
    dec_init = np.vstack([x_init.reshape(-1,1), u_init.reshape(-1,1)])
    # bounds for g (dynamics equality + terminal constraint)
    ng = nx * (N + 1) + 1 # nx * (N+1) for dynamics, +1 for terminal constraint
    lbg = np.zeros((ng,1))
    ubg = np.zeros((ng,1))
    # The terminal constraint (x_N - xbar)' P (x_N - xbar) - α <= 0 means the upper bound for this constraint is 0
    lbg[-1] = -ca.inf  # allow anything below 0
    ubg[-1] = 0        # enforce (x_N - xbar)^T P (x_N - xbar) - α <= 0

    # bounds for dec vars: we won't impose here; instead we later clip applied input
    p_val = np.vstack([x0_val.reshape(-1,1), xbar_val.reshape(-1,1), ubar_val.reshape(-1,1)])
    try:
        sol = solver(x0=dec_init, lbg=lbg, ubg=ubg, p=p_val)
        dec_sol = sol['x'].full().flatten()
        # extract first control
        x_vars = dec_sol[:nx*(N+1)].reshape((nx, N+1))
        u_vars = dec_sol[nx*(N+1):].reshape((nu, N))
        u0 = u_vars[:,0]

        # Terminal constraint check (optional, for debugging/info)
        xN = x_vars[:, -1]
        x_diff = xN - xbar_val
        V_terminal = x_diff.T @ P @ x_diff
        if V_terminal > alpha + 1e-6: # Add tolerance for numerical issues
            print(f"Warning: terminal constraint violated at step {{k}}. V = {{float(V_terminal):.4f}}, alpha = {{float(alpha):.4f}}")

    except Exception as e:
        # infeasible or solver failure -> fallback to zero accel
        print("MPC solver failed:", e)
        u0 = np.zeros(2)
    # clip inputs
    u0 = np.clip(u0, -max_acc, max_acc)
    return u0


# -------------------------
# Main sim initialization and Simulation loop (Algorithm 1 with Algo 2 terminal ingredients)
# -------------------------
if __name__ == "__main__":
    np.random.seed(2)
    # initial robot states: random positions, zero velocity
    x = np.zeros((M, 4))
    #x[:, 0:2] = np.random.rand(M, 2) * np.array([Lx, Ly])
    # x[:, 0:2] = np.array([[0.51, 0.51], [0.49, 0.49], [0.49, 0.51], [0.51, 0.49]])
    x[:, 0:2] = np.array([[0.05,0.15],  [0.1,0.15],  [0.25,0.35],  [0.3,0.1]])

    # initial references (centroids) computed from initial Voronoi
    ps = x[:, :2].copy()
    cells = bounded_voronoi(ps, Q_region)
    centroids = np.zeros((M,2))
    mass_list = np.zeros(M)
    for i in range(M):
        c, mass = weighted_centroid_of_polygon(cells[i], phi_uniform, samples_per_side=grid_density)
        if mass == 0:
            c = ps[i]
        centroids[i] = c
        mass_list[i] = mass
    # per-agent tracking error initial
    e_r = np.linalg.norm(ps - centroids, axis=1)

    # logging
    traj = [ps.copy()]
    H_vals = []
    # Build MPC solver with terminal ingredients
    mpc_solver, terminal_ingredients = build_mpc_problem()
    P_term = terminal_ingredients['P']
    K_term = terminal_ingredients['K']
    alpha_term = terminal_ingredients['alpha']

    x_sym = ca.SX.sym('x_sym', nx)
    u_sym = ca.SX.sym('u_sym', nu)
    # f_dynamics = ca.Function('f_dyn', [x_sym, u_sym], [dynamic_model(x_sym, u_sym)]) # CasADi function needed if using CasADi version of dynamic_model

    print("Built MPC solver with terminal ingredients:")
    print(f"  alpha = {float(alpha_term):.4f}")
    # print("
 P = ", P_term) # Keep commented unless needed for debugging due to large output
    # print("
 K = ", K_term) # Keep commented unless needed for debugging due to large output
    print("Initial Position:", x[:, 0:2])
    print("Starting simulation...")

    start_time = time.time()
    for k in range(sim_steps):
        # compute Voronoi and centroids (based on current positions)
        ps = x[:, :2].copy()
        cells = bounded_voronoi(ps, Q_region)
        centroids_current = np.zeros((M,2))
        masses = np.zeros(M)
        for i in range(M):
            c, mass = weighted_centroid_of_polygon(cells[i], phi_uniform, samples_per_side=grid_density)
            if mass == 0:
                c = ps[i]
            centroids_current[i] = c
            masses[i] = mass

        # for each agent: compute MPC to track its current reference ri (which is centroids[i] from last update)
        # The algorithm in the paper sets ri to computed centroid at initialization, then only updates ri when condition met.
        # To mimic that, we will store ri, ei,r, and update ri only when condition satisfied.
        # For start: if first iteration, we set ri as previously computed centroids (done above at init)
        if k == 0:
            ri = centroids.copy()
        # apply MPC for each agent to track ri[i]
        for i in range(M):
            x0 = x[i].copy()
            # build xbar (steady state) from ri: we choose steady state with zero velocity, so xbar = [rx, ry, 0,0]
            xbar = np.array([ri[i,0], ri[i,1], 0., 0.])
            ubar = np.zeros(2)
            # solve MPC
            u0 = solve_mpc_casadi(mpc_solver, P_term, alpha_term, x0, xbar, ubar)
            # apply input, propagate dynamics
            # Use the numpy version of dynamic model here if not using CasADi function object
            x_next = dynamic_model(x0, u0) # Assuming dynamic_model is available and works with numpy arrays
            x_next = np.array(x_next).flatten() # Ensure it's a numpy array
            x_next = clip_state(x_next)
            x[i] = x_next
            #if k <= 10 and i == 1:
              #print("x0 =", x0, "; u0 =", u0, "; x_next =", x_next, "; xbar =", xbar)
        # After all agents applied first input, compute condition for updating ri (lines 8-10)
        ps_new = x[:, :2].copy()
        # check not-increased condition
        dist_to_ri = np.linalg.norm(ps_new - ri, axis=1)
        non_increase = np.all(dist_to_ri <= e_r + 1e-9)
        decreased_any = np.any(dist_to_ri < e_r - 1e-9)
        if non_increase and decreased_any:
            # update ri to new centroids computed from current positions (centroids_current)
            ri = centroids_current.copy()
            e_r = np.linalg.norm(ps_new - ri, axis=1)
            # (this is centralized step in algorithm)
        else:
            # keep e_r as distances to current ri if not updated
            e_r = dist_to_ri.copy()

        # log
        traj.append(x[:, :2].copy())
        # Assuming bounded_voronoi, phi_uniform, Point are available
        H_vals.append(approx_coverage_cost(ps_new, cells, phi_uniform, gfun=lambda d: d**2, samples=400))
        # stop if robots near centroids (converged)
        if np.max(np.linalg.norm(ps_new - centroids_current, axis=1)) < 0.05:
            print(f"Converged by iteration {k}")
            break

    end_time = time.time()
    print("Simulation finished in {:.2f}s".format(end_time - start_time))

    # -------------------------
    # Visualization
    # -------------------------
    traj_np = np.array(traj)  # (T, M, 2)
    T_sim = traj_np.shape[0]

    plt.figure(figsize=(8,8))
    # plot workspace
    plt.xlim(0-0.01, Lx+0.01); plt.ylim(0-0.01, Ly+0.01)
    # plot trajectories
    for i in range(M):
        plt.plot(traj_np[:, i, 0], traj_np[:, i, 1], '-', linewidth=1)
        plt.scatter(traj_np[0, i, 0], traj_np[0, i, 1], marker='o', label=f'robot{i}' if i==0 else None)
        plt.scatter(traj[-1, i, 0], traj[-1, i, 1], marker='x')
    # plot final Voronoi
    final_cells = bounded_voronoi(traj_np[-1,:,:], Q_region)
    for c in final_cells:
        if not c.is_empty:
            xs, ys = c.exterior.xy
            plt.plot(xs, ys, 'k--', linewidth=0.7)
    # final centroids
    final_centroids = np.array([weighted_centroid_of_polygon(c, phi_uniform, grid_density)[0] for c in final_cells])
    plt.scatter(final_centroids[:,0], final_centroids[:,1], marker='*', s=120, c='r', label='centroids')
    plt.title('Trajectories and final Voronoi')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    plt.figure()
    plt.plot(H_vals)
    plt.title('Approximate coverage cost H(p,W) over time')
    plt.xlabel('iteration')
    plt.ylabel('H')
    plt.grid(True)
    plt.show()
