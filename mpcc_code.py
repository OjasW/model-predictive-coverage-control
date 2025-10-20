# -*- coding: utf-8 -*-
"""
@author: wanio
"""
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import polygonize, unary_union
from scipy.linalg import solve_discrete_are
import time

"""Parameters"""

Lx, Ly = 1.0, 1.0           # workspace size
Q_region = box(0, 0, Lx, Ly)
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




# Voronoi Partitions and Centroids
def bounded_voronoi(polypoints, bbox):
    """
    Compute bounded Voronoi cells for points inside bbox.
    Returns list of shapely Polygons of length = len(polypoints).
    """
    # scipy Voronoi unbounded; clip cells to bounding box
    pts = np.asarray(polypoints)
    if len(pts) == 1:
        return [bbox]
    vor = Voronoi(pts)
    # build polygon for each region
    regions = []
    for i, point in enumerate(pts):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if -1 in region:  # unbounded region -> clip with bbox
            # approximate by taking voronoi ridges and intersecting rays with bbox edges
            # easier: build large polygon from ridges, then intersect with bbox
            # We'll reconstruct the cell as intersection of half-planes: for each other point j,
            # the halfplane { x | ||x - p_i|| <= ||x - p_j|| }.
            cell = bbox
            pi = point
            for j, pj in enumerate(pts):
                if j == i: continue
                # perpendicular bisector between pi and pj: line ax+by+c=0
                mid = 0.5*(pi + pj)
                # normal vector from pj to pi
                n = np.array(pi) - np.array(pj)
                if np.linalg.norm(n) < 1e-8:
                    continue
                # define a large polygon half-space via a line
                # build a line across bbox and take half that contains pi
                # create a long line segment orthogonal to n passing near mid
                dir_vec = np.array([-n[1], n[0]])
                p1 = mid + dir_vec * max(Lx, Ly) * 10
                p2 = mid - dir_vec * max(Lx, Ly) * 10
                half_line = LineString([tuple(p1), tuple(p2)])
                # create large polygon for half-plane
                coords = [tuple(p1), tuple(p2),
                          (p2[0] + n[0]*1e6, p2[1] + n[1]*1e6),
                          (p1[0] + n[0]*1e6, p1[1] + n[1]*1e6)]
                half_poly = Polygon(coords)
                # choose which half contains pi
                if not half_poly.contains(Point(tuple(pi))):
                    half_poly = Polygon(list(half_poly.exterior.coords)[::-1])
                cell = cell.intersection(half_poly)
                if cell.is_empty:
                    break
            regions.append(cell)
        else:
            polygon = Polygon([vor.vertices[v] for v in region])
            regions.append(polygon.intersection(bbox))
    # ensure order corresponds to input points
    # convert possible GeometryCollections to Polygons
    final = []
    for g in regions:
        if g.geom_type == 'Polygon':
            final.append(g)
        elif g.is_empty:
            final.append(Polygon())
        else:
            # unify and take convex hull fallback
            try:
                gp = unary_union(g)
                final.append(gp.convex_hull.intersection(bbox))
            except Exception:
                final.append(Polygon())
    return final

# -------------------------
# Sensor density (Phi)
# -------------------------
def phi_uniform(xy):
    # uniform density (constant)
    return 1.0

def weighted_centroid_of_polygon(poly, phi=phi_uniform, samples_per_side=20):
    """
    Numerically estimate weighted centroid over polygon using grid sampling.
    Returns centroid (x,y) and mass.
    """
    if poly.is_empty or not poly.is_valid:
        return np.array([np.nan, np.nan]), 0.0
    minx, miny, maxx, maxy = poly.bounds
    if minx == maxx or miny == maxy:
        # degenerate cell
        pts = np.array(poly.exterior.coords)
        c = np.mean(pts, axis=0)
        return c, 1.0
    nx = samples_per_side
    ny = samples_per_side
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    mask = np.array([poly.contains(Point(p)) or poly.touches(Point(p)) for p in pts])
    pts_in = pts[mask]
    if pts_in.shape[0] == 0:
        # fallback centroid of polygon geometry
        c = np.array(poly.representative_point().coords[0])
        return c, 1.0
    vals = np.array([phi(p) for p in pts_in])
    mass = vals.sum()
    if mass == 0:
        c = pts_in.mean(axis=0)
        return c, 0.0
    centroid = (vals[:, None] * pts_in).sum(axis=0) / mass
    return centroid, mass


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


# Keeping the robot within the bounding box
def clip_state(x):
    # position inside Q, velocity limits
    x = x.copy()
    x[0] = np.clip(x[0], 0, Lx)
    x[1] = np.clip(x[1], 0, Ly)
    x[2] = np.clip(x[2], -max_vel[0], max_vel[0])
    x[3] = np.clip(x[3], -max_vel[1], max_vel[1])
    return x


# small helper for coverage cost approx H(p,W) using Monte Carlo grid
def approx_coverage_cost(points, cells, phi=phi_uniform, gfun=lambda d: d**2, samples=400):
    # approximate integral by sampling a grid over workspace
    xs = np.linspace(0, Lx, int(np.sqrt(samples)))
    ys = np.linspace(0, Ly, int(np.sqrt(samples)))
    X, Y = np.meshgrid(xs, ys)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    total = 0.0
    for p_i, cell in zip(points, cells):
        # pick samples inside cell
        mask = np.array([cell.contains(Point(pt)) or cell.touches(Point(pt)) for pt in pts])
        if mask.sum() == 0:
            continue
        pts_in = pts[mask]
        dists = np.linalg.norm(pts_in - p_i, axis=1)
        vals = gfun(dists) * np.array([phi(pt) for pt in pts_in])
        total += vals.mean() * cell.area
    return total

"""Algorithm 1 and 2"""

# Cost weights
Q_stage = np.diag([1.9, 1.9, 0.02, 0.01])   # state error weight
R_stage = np.diag([0.08, 0.02])               # input weight


def linearize_dynamics(xbar, ubar, nx=nx, nu=nu):
    """
    Compute linearization A,B of f around (xbar, ubar).
    """
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
    """Discrete LQR via solve_discrete_are. Returns K,P (numpy)."""
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    P = 0.5 * (P + P.T)  # ensure symmetry
    return K, P

MPC Builder With Terminal LQR

# @title
# Build MPC
def build_mpc_problem(nx=nx, nu=nu, N=N_mpc, Q_stage=Q_stage, R_stage=R_stage,
                      xbar_lin=None, ubar_lin=None,
                      u_min=-max_acc, u_max=max_acc):
    
    xbar_lin = np.zeros(nx)
    ubar_lin = np.zeros(nu)
    xbar_lin[2] = 0.1  # Small non-zero velocity in x direction
    ubar_lin[0] = 0.1  # Small non-zero velocity
    # A, B
    X_lin = ca.SX.sym('X', nx)
    U_lin = ca.SX.sym('U', nu)
    A_lin, B_lin = linearize_dynamics(xbar_lin, ubar_lin, nx=nx, nu=nu)

    print("A matrix before dlqr:", A_lin)
    print("B matrix before dlqr:", B_lin)

    # K, P
    K, P = dlqr(A_lin, B_lin, Q_stage, R_stage)
    # alpha
    invP = np.linalg.inv(P)
    quad_vals = np.array([K[j:j+1, :] @ invP @ K[j:j+1, :].T for j in range(nu)]).flatten()
    alpha1_candidates = []
    for j in range(nu):
        if quad_vals[j] <= 0:
            alpha1_candidates.append(np.inf)
        else:
            alpha1_candidates.append((u_max[j]**2) / float(quad_vals[j]))
    alpha1 = min(alpha1_candidates)
    alpha = 0.5 * alpha1  

    # Build MPC
    X = ca.SX.sym('X', nx, N+1)
    U = ca.SX.sym('U', nu, N)
    x0 = ca.SX.sym('x0', nx)
    xbar = ca.SX.sym('xbar', nx) 
    ubar = ca.SX.sym('ubar', nu)
    # stage cost matrices as CasADi
    Qc = ca.DM(Q_stage)
    Rc = ca.DM(R_stage)
    QN = ca.DM(P)
    obj = 0
    g = []
    # initial cond
    g.append(X[:,0] - x0)
    for k in range(N):
        x_next = dynamic_model(X[:,k], U[:,k])
        g.append(X[:,k+1] - x_next)
        dx = X[:,k] - xbar
        du = U[:,k] - ubar
        obj = obj + ca.mtimes([dx.T, Qc, dx]) + ca.mtimes([du.T, Rc, du])
    # terminal cost
    dxN = X[:,N] - xbar
    obj = obj + ca.mtimes([dxN.T, QN, dxN])
    # Adding terminal constraint x_N' P x_N - α <= 0
    g.append(ca.mtimes([X[:, N].T, QN, X[:, N]]) - float(alpha))
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

# Solve MPC
def solve_mpc_casadi(solver, P, alpha, x0_val, xbar_val, ubar_val, nx=nx, nu=nu, N=N_mpc):
    
    # initial guess
    x_init = np.tile(x0_val.reshape(-1,1), (1, N+1))
    u_init = np.zeros((nu, N))
    dec_init = np.vstack([x_init.reshape(-1,1), u_init.reshape(-1,1)])
    # bounds for g (dynamics equality)
    # Get the number of constraints from the solver object
    ng = nx * (N + 1) + 1
    lbg = np.zeros((ng,1))
    ubg = np.zeros((ng,1))
    # The terminal constraint x_N' P x_N - α <= 0 means the upper bound for this constraint is 0
    lbg[-1] = -np.inf  # allow anything below 0
    ubg[-1] = 0        # enforce x_N^T P x_N - α <= 0
    # bounds for dec vars: we won't impose here; instead we later clip applied input
    p_val = np.vstack([x0_val.reshape(-1,1), xbar_val.reshape(-1,1), ubar_val.reshape(-1,1)])
    try:
        sol = solver(x0=dec_init, lbg=lbg, ubg=ubg, p=p_val)
        dec_sol = sol['x'].full().flatten()
        # extract first control
        x_vars = dec_sol[:nx*(N+1)].reshape((nx, N+1))
        u_vars = dec_sol[nx*(N+1):].reshape((nu, N))
        u0 = u_vars[:,0]

        # Terminal constraint check
        xN = x_vars[:, -1]
        x_diff = xN - xbar_val
        V_terminal = x_diff.T @ P @ x_diff
        if V_terminal > alpha:
            print("Warning: terminal constraint violated. V =", float(V_terminal))

    except Exception as e:
        # infeasible or solver failure -> fallback to zero accel
        print("MPC solver failed:", e)
        u0 = np.zeros(2)
    # clip inputs
    u0 = np.clip(u0, -max_acc, max_acc)
    return u0


#Initialize Simulation
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
mpc_solver, terminal = build_mpc_problem()
x_sym = ca.SX.sym('x_sym', nx)
u_sym = ca.SX.sym('u_sym', nu)
f_dynamics = ca.Function('f_dyn', [x_sym, u_sym], [dynamic_model(x_sym, u_sym)])
P_term = terminal['P']
K_term = terminal['K']
alpha_term = terminal['alpha']
print("Built MPC solver with terminal ingredients:")
print("  alpha =", alpha_term, "\n P = ", P_term, "\n K = ", K_term)
print("Initial Position:", x[:, 0:2])
print("Simulation Loop ready to be run.")


# Simulation Loop
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
        x_next = f_dynamics(x0, u0).full().flatten()
        x_next = clip_state(x_next)
        x[i] = x_next
        
    # After all agents applied first input, compute condition for updating ri 
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
    H_vals.append(approx_coverage_cost(ps_new, cells, phi_uniform, gfun=lambda d: d**2, samples=400))
    # stop if robots near centroids (converged)
    if np.max(np.linalg.norm(ps_new - centroids_current, axis=1)) < 0.05:
        print(f"Converged by iteration {k}")
        break

end_time = time.time()
print("Simulation finished in {:.2f}s".format(end_time - start_time))


# Graphs
traj = np.array(traj)  # (T, M, 2)
T = traj.shape[0]

plt.figure(figsize=(8,8))
# plot workspace
plt.xlim(0-0.01, Lx+0.01); plt.ylim(0-0.01, Ly+0.01)
# plot trajectories
for i in range(M):
    plt.plot(traj[:, i, 0], traj[:, i, 1], '-', linewidth=1)
    plt.scatter(traj[0, i, 0], traj[0, i, 1], marker='o', label=f'robot{i}' if i==0 else None)
    plt.scatter(traj[-1, i, 0], traj[-1, i, 1], marker='x')
# plot final Voronoi
final_cells = bounded_voronoi(traj[-1,:,:], Q_region)
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