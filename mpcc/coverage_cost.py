
import numpy as np
from shapely.geometry import Point, Polygon

# Assuming Lx, Ly, and phi_uniform are defined globally or passed as arguments
# For this script, we'll keep them as in the original cell, assuming they are defined elsewhere.
# Let's define them here for the script to be runnable standalone, adjust as needed.
Lx, Ly = 1.0, 1.0 # Default values, replace with actual if needed

def phi_uniform(xy):
    # uniform density (constant)
    return 1.0

# small helper for coverage cost approx H(p,W) using Monte Carlo grid
def approx_coverage_cost(points, cells, phi=phi_uniform, gfun=lambda d: d**2, samples=400):
    # approximate integral by sampling a grid over workspace
    # Need Lx and Ly defined or passed
    # Need Point defined from shapely.geometry
    # Need cells to be shapely Polygons

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
