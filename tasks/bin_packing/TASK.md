# Online bin packing

Items arrive one at a time and must go into a bin of capacity 100 at once, before the next
item is seen. `priority.py` defines `priority(item, bins)`: `item` is the size of the arriving
item and `bins` is a numpy array with the remaining capacity of every bin that can take it,
partly filled and empty ones alike. One empty bin per item is available from the start, so
putting the item into a bin with remaining capacity 100 opens a new bin. Return a numpy array
of the same length, one score per bin: the item goes into the bin with the highest score.

The judge packs 5 fixed instances of 5,000 items whose sizes follow a Weibull distribution
(shape 3, scale 45, rounded and clipped to 1..100). Each instance scores 100 × (lower bound on
the number of bins) / (bins used), so no packing can score above 100; the starting candidate,
best fit, scores 95.94. The score is the mean over the instances. A crash, an array of the
wrong shape, or more than 60 seconds for all five instances makes the candidate invalid.

Python 3.11 and numpy are available. Do not read or write files or use the network.
