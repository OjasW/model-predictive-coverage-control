
import numpy as np
from shapely.geometry import Polygon, Point

# Assuming Lx, Ly, and phi_uniform are defined globally or passed as arguments
# For this script, we'll keep them as in the original cell, assuming they are defined elsewhere.
# Let's define them here for the script to be runnable standalone, adjust as needed.
Lx, Ly = 1.0, 1.0 # Default values, replace with actual if needed

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
