"""Scores packing.py. `pack()` runs in a fresh process; this judge only checks its output.
Run with `python -P` so that the candidate's files cannot shadow modules imported here."""

import json
import math
import os
import subprocess
import sys

N, TOL, LIMIT = 26, 1e-9, 30


def fail(message):
    print(json.dumps({"score": None, "feedback": message}))
    sys.exit(0)


def main():
    try:
        proc = subprocess.run([sys.executable, "-P", __file__, "--pack"], capture_output=True, text=True,
                              timeout=LIMIT)
    except subprocess.TimeoutExpired:
        fail(f"pack() did not return within {LIMIT}s")
    if proc.returncode:
        fail(f"pack() crashed:\n{proc.stderr[-1500:]}")
    try:
        circles = [tuple(float(v) for v in c) for c in json.loads(proc.stdout.strip().splitlines()[-1])]
    except (ValueError, IndexError, TypeError):
        fail("pack() must return a list of (x, y, r) tuples")
    if len(circles) != N or any(len(c) != 3 for c in circles):
        fail(f"expected {N} circles of 3 numbers each, got {len(circles)}")
    problems = []
    for i, (x, y, r) in enumerate(circles):
        if not all(map(math.isfinite, (x, y, r))) or r <= 0:
            problems.append(f"circle {i} has a non-finite value or a non-positive radius")
        elif min(x - r, y - r, 1 - x - r, 1 - y - r) < -TOL:
            problems.append(f"circle {i} sticks out of the square by {-min(x - r, y - r, 1 - x - r, 1 - y - r):.2e}")
    for i in range(N):
        for j in range(i + 1, N):
            (xi, yi, ri), (xj, yj, rj) = circles[i], circles[j]
            gap = math.hypot(xi - xj, yi - yj) - ri - rj
            if gap < -TOL:
                problems.append(f"circles {i} and {j} overlap by {-gap:.2e}")
    if problems:
        fail(f"{len(problems)} violations, e.g.:\n" + "\n".join(problems[:8]))
    radii = sorted(r for _, _, r in circles)
    total = sum(radii)
    print(json.dumps({"score": round(total, 9),
                      "feedback": f"valid; sum of radii {total:.6f}; radii from {radii[0]:.4f} to {radii[-1]:.4f}"}))


def worker():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # after site-packages: no shadowing
    from packing import pack

    print("\n" + json.dumps([[float(v) for v in c] for c in pack()]))


if __name__ == "__main__":
    worker() if sys.argv[1:] == ["--pack"] else main()
