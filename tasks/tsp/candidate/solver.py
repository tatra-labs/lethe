import numpy as np


def solve(points, time_limit):
    """Nearest-neighbor tour from city 0."""
    pts = np.asarray(points)
    n = len(pts)
    unvisited = np.ones(n, dtype=bool)
    tour = [0]
    unvisited[0] = False
    for _ in range(n - 1):
        d = np.linalg.norm(pts - pts[tour[-1]], axis=1)
        d[~unvisited] = np.inf
        nxt = int(np.argmin(d))
        tour.append(nxt)
        unvisited[nxt] = False
    return tour
