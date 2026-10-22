# VLIW SIMD kernel (Anthropic's original performance take-home)

`perf_takehome.py` builds a program for the simulated VLIW SIMD machine defined in
`problem.py`. `KernelBuilder.build_kernel(forest_height, n_nodes, batch_size, rounds)` must
emit instruction bundles that traverse the tree and hash the batch of values exactly as
`reference_kernel2` in `problem.py` does. Read `problem.py` for the machine: its engines,
slot limits per cycle, instructions, scratch space and memory layout.

Minimize the number of clock cycles the machine takes. The judge builds your kernel once,
for forest_height=10, batch_size=256 and rounds=16, in a separate process. It then runs the
resulting program on a frozen, unmodified copy of the simulator against 4 random inputs.
Any wrong output makes the candidate invalid. The score is the cycle count; the starting
kernel takes 147,734 cycles.

Editing `problem.py` changes nothing for the judge, which uses its own copy, and multicore
is disabled on purpose. For reference, Claude Opus 4.5 reached 1,487 cycles after 11.5
hours in Anthropic's test-time compute harness.
