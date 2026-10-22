"""Scores solver.py. Each instance is solved in a fresh process, timed from outside, so a
candidate can tamper with neither the clock nor this judge. Run with `python -P` so that
files in the candidate's directory can never shadow the modules imported here."""

import json
import os
import subprocess
import sys
import time

import numpy as np

SIZES = [200, 400, 600, 800, 1000, 1300, 1600, 2000]
TIME_LIMIT = 2.0
HARD_LIMIT = 3.0  # includes interpreter start-up


def instance(k, n):
    rng = np.random.default_rng(1000 + k)
    if k % 2:  # clustered
        centers = rng.random((max(4, n // 100), 2))
        pts = centers[rng.integers(len(centers), size=n)] + rng.normal(0.0, 0.04, (n, 2))
        return np.clip(pts, 0.0, 1.0)
    return rng.random((n, 2))


def length(pts, tour):
    p = pts[tour]
    return float(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum())


def nearest_neighbor(pts):
    n = len(pts)
    free = np.ones(n, dtype=bool)
    tour, free[0] = [0], False
    for _ in range(n - 1):
        d = np.linalg.norm(pts - pts[tour[-1]], axis=1)
        d[~free] = np.inf
        tour.append(int(np.argmin(d)))
        free[tour[-1]] = False
    return tour


def fail(message):
    print(json.dumps({"score": None, "feedback": message}))
    sys.exit(0)


def main():
    scores, notes = [], []
    for k, n in enumerate(SIZES):
        pts = instance(k, n)
        payload = json.dumps({"points": pts.tolist(), "time_limit": TIME_LIMIT})
        start = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-P", __file__, "--solve"], input=payload,
                                  capture_output=True, text=True, timeout=HARD_LIMIT + 2)
        except subprocess.TimeoutExpired:
            fail(f"instance {k} (n={n}): no answer within {HARD_LIMIT:.0f}s")
        seconds = time.perf_counter() - start
        if proc.returncode:
            fail(f"instance {k} (n={n}): solve crashed:\n{proc.stderr[-1500:]}")
        if seconds > HARD_LIMIT:
            fail(f"instance {k} (n={n}): took {seconds:.2f}s, over the {HARD_LIMIT:.0f}s limit")
        try:
            tour = [int(c) for c in json.loads(proc.stdout.strip().splitlines()[-1])]
        except (ValueError, IndexError, TypeError):
            fail(f"instance {k} (n={n}): solve did not return a list of city indices")
        if sorted(tour) != list(range(n)):
            fail(f"instance {k} (n={n}): the tour must visit each of the {n} cities exactly once")
        score = 100.0 * length(pts, nearest_neighbor(pts)) / length(pts, tour)
        scores.append(score)
        notes.append(f"n={n} {'clustered' if k % 2 else 'uniform'}: {score:.2f} in {seconds:.2f}s")
    print(json.dumps({"score": round(float(np.mean(scores)), 4), "feedback": "\n".join(notes)}))


def worker():
    data = json.load(sys.stdin)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # after site-packages: no shadowing
    from solver import solve

    tour = solve([tuple(p) for p in data["points"]], data["time_limit"])
    print("\n" + json.dumps([int(c) for c in tour]))


if __name__ == "__main__":
    worker() if sys.argv[1:] == ["--solve"] else main()
