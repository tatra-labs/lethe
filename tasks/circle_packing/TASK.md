# Circle packing

`packing.py` must define `pack()`, which returns 26 circles as `(x, y, r)` tuples. Every
circle must lie inside the unit square [0, 1] × [0, 1] and no two circles may overlap;
touching is fine, and violations up to 1e-9 are tolerated. Maximize the sum of the radii.

The judge calls `pack()` once in a fresh process and allows it 30 seconds. The score is the
sum of the radii; any violation makes the candidate invalid. The starting candidate scores
1.3. The best packings known reach about 2.6359.

Python 3.11, numpy and scipy are available. Do not read or write files or use the network.
