
import numpy as np
import casadi as ca
import time
import matplotlib.pyplot as plt
from shapely.geometry import box, Point, Polygon

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
Q_stage = np.diag([2, 2, 0.02, 0.01])   # state error weight
R_stage = np.diag([0.08, 0.02]) #1 * np.eye(2)           # input weight
Q_terminal = np.diag([5, 30, 0.15, 2.5])

# -------------------------
# MPC Problem Builder
# -------------------------
def build_mpc_problem(nx=nx, nu=nu, N=N_mpc, Q_stage=Q_stage, R_stage=R_stage, Q_terminal=Q_terminal):
    X = ca.SX.sym('X', nx, N+1)
    U = ca.SX.sym('U', nu, N)
    x0 = ca.SX.sym('x0', nx)
    xbar = ca.SX.sym('xbar', nx)  # steady state to track
    ubar = ca.SX.sym('ubar', nu)
    # stage cost matrices as CasADi
    Qc = ca.DM(Q_stage)
    Rc = ca.DM(R_stage)
    QN = ca.DM(Q_terminal)
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
    # bounds and constraints will be added in solver call (as functions)
    # pack decision vars
    dec_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    # Pack NLP
    g_all = ca.vertcat(*g)
    p = ca.vertcat(x0, xbar, ubar)
    nlp = {'x': dec_vars, 'f': obj, 'g': g_all, 'p': p}
    # Create solver
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    return solver

# -------------------------
# MPC Solver
# -------------------------
def solve_mpc_casadi(solver, x0_val, xbar_val, ubar_val, nx=nx, nu=nu, N=N_mpc, max_acc=max_acc):
    # initial guess
    x_init = np.tile(x0_val.reshape(-1,1), (1, N+1))
    u_init = np.zeros((nu, N))
    dec_init = np.vstack([x_init.reshape(-1,1), u_init.reshape(-1,1)])
    # bounds for g (dynamics equality)
    ng = nx*(N+1)
    lbg = np.zeros((ng,1))
    ubg = np.zeros((ng,1))
    # bounds for dec vars: we won't impose here; instead we later clip applied input
    p_val = np.vstack([x0_val.reshape(-1,1), xbar_val.reshape(-1,1), ubar_val.reshape(-1,1)])
    try:
        sol = solver(x0=dec_init, lbg=lbg, ubg=ubg, p=p_val)
        dec_sol = sol['x'].full().flatten()
        # extract first control
        x_vars = dec_sol[:nx*(N+1)].reshape((nx, N+1))
        u_vars = dec_sol[nx*(N+1):].reshape((nu, N))
        u0 = u_vars[:,0]
    except Exception as e:
        # infeasible or solver failure -> fallback to zero accel
        print("MPC solver failed:", e)
        u0 = np.zeros(2)
    # clip inputs
    u0 = np.clip(u0, -max_acc, max_acc)
    return u0


# -------------------------
# Main sim initialization and Simulation loop (Algorithm 1)
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
    mpc_solver = build_mpc_problem()
    x_sym = ca.SX.sym('x_sym', nx)
    u_sym = ca.SX.sym('u_sym', nu)
    # f_dynamics = ca.Function('f_dyn', [x_sym, u_sym], [dynamic_model(x_sym, u_sym)]) # CasADi function needed if using CasADi version of dynamic_model

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
            u0 = solve_mpc_casadi(mpc_solver, x0, xbar, ubar)
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
    # Visualization (Optional - can be done in a separate script/notebook)
    # -------------------------
    # If you want to include visualization, you'll need matplotlib and the animation code.
    # For a basic file, we'll just print results or save data.
    # Example: Save trajectory and H_vals to files for plotting elsewhere
    # np.save('traj.npy', np.array(traj))
    # np.save('H_vals.npy', np.array(H_vals))

    # Or include basic plotting if matplotlib is installed
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
