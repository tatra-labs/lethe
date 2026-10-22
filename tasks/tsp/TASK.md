# Traveling salesman

`solver.py` must define `solve(points, time_limit)`. `points` is a list of `(x, y)` city
coordinates in the unit square and `time_limit` is a number of seconds. Return a tour: a
list that holds every city index exactly once. The tour returns to its first city.

The judge runs `solve` on 8 fixed instances of 200 to 2,000 cities, half spread uniformly
and half in clusters, with `time_limit=2.0`. A call that takes longer than 3 seconds,
including the start-up of a fresh Python process, fails. Each instance scores
100 × (length of the nearest-neighbor tour from city 0) / (length of your tour), so the
starting candidate scores 100 and shorter tours score more. The score is the mean over the
instances. A failure or an invalid tour on any instance makes the whole candidate invalid.

Python 3.11 and numpy are available. Do not read or write files or use the network.
