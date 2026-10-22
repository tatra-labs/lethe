"""Scores priority.py on online bin packing, with FunSearch's evaluation protocol: one bin per item
is allocated up front, and `priority` scores every bin that can take the item, empty ones included.

The heuristic runs in a separate process, which reports the bin chosen for every item. This judge
replays those choices, checks every capacity and counts the bins itself. Run with `python -P` so
that the candidate's files cannot shadow the modules imported here.
"""

import json
import math
import os
import subprocess
import sys

import numpy as np

CAPACITY, ITEMS, INSTANCES, LIMIT = 100, 5000, 5, 60


def instance(k):
    rng = np.random.default_rng(2024 + k)
    return np.clip(np.round(rng.weibull(3.0, ITEMS) * 45), 1, CAPACITY).astype(int).tolist()


def fail(message):
    print(json.dumps({"score": None, "feedback": message}))
    sys.exit(0)


def main():
    instances = [instance(k) for k in range(INSTANCES)]
    try:
        proc = subprocess.run([sys.executable, "-P", __file__, "--pack"], input=json.dumps(instances),
                              capture_output=True, text=True, timeout=LIMIT)
    except subprocess.TimeoutExpired:
        fail(f"packing the {INSTANCES} instances took more than {LIMIT}s")
    if proc.returncode:
        fail(f"priority crashed:\n{proc.stderr[-1500:]}")
    try:
        choices = json.loads(proc.stdout.strip().splitlines()[-1])
        assert len(choices) == INSTANCES
    except (ValueError, IndexError, AssertionError):
        fail("could not read the packing that priority produced")

    scores, notes = [], []
    for k, (items, chosen) in enumerate(zip(instances, choices)):
        if len(chosen) != len(items):
            fail(f"instance {k}: {len(chosen)} choices for {len(items)} items")
        room = [CAPACITY] * len(items)  # remaining capacity of every bin
        for item, b in zip(items, chosen):
            if not isinstance(b, int) or not 0 <= b < len(room):
                fail(f"instance {k}: invalid bin index {b!r}")
            if room[b] < item:
                fail(f"instance {k}: an item of size {item} does not fit in bin {b}")
            room[b] -= item
        used = sum(r < CAPACITY for r in room)
        bound = math.ceil(sum(items) / CAPACITY)
        scores.append(100 * bound / used)
        notes.append(f"instance {k}: {used} bins, lower bound {bound}, score {scores[-1]:.3f}")
    print(json.dumps({"score": round(float(np.mean(scores)), 4), "feedback": "\n".join(notes)}))


def worker():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # after site-packages: no shadowing
    from priority import priority

    out = []
    for items in json.load(sys.stdin):
        room, chosen = np.full(len(items), float(CAPACITY)), []
        for item in items:
            fits = np.flatnonzero(room >= item)
            scores = np.asarray(priority(float(item), room[fits].copy()), dtype=float)
            if scores.shape != (len(fits),):
                raise ValueError(f"priority returned shape {scores.shape} for {len(fits)} bins")
            b = int(fits[int(np.argmax(scores))])
            room[b] -= item
            chosen.append(b)
        out.append(chosen)
    print("\n" + json.dumps(out))


if __name__ == "__main__":
    worker() if sys.argv[1:] == ["--pack"] else main()
