
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import polygonize, unary_union

# Assuming Lx and Ly are defined globally or passed as arguments if needed
# For now, we'll assume they are available in the environment where this function is used.
# Or you can pass them as arguments to the function if preferred.
# Example: def bounded_voronoi(polypoints, bbox, Lx, Ly):
# For this script, we'll keep them as in the original cell, assuming they are defined elsewhere.
# Let's define them here for the script to be runnable standalone, adjust as needed.
Lx, Ly = 1.0, 1.0 # Default values, replace with actual if needed


def bounded_voronoi(polypoints, bbox):
    """
    Compute bounded Voronoi cells for points inside bbox.
    Returns list of shapely Polygons of length = len(polypoints).
    """
    # scipy Voronoi unbounded; we clip cells to bounding box
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

