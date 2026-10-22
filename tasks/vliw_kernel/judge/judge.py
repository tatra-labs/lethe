"""Scores perf_takehome.py on a frozen copy of the simulator.

The kernel is built in a separate process and only its instruction list comes back, so
candidate code never runs next to the simulator that counts the cycles. Run with
`python -P` so that the candidate's files cannot shadow modules imported here.
"""

import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HEIGHT, BATCH, ROUNDS = 10, 256, 16
SEEDS = [11, 22, 33, 44]
BASELINE = 147734


def fail(message):
    print(json.dumps({"score": None, "feedback": message}))
    sys.exit(0)


def main():
    sys.path.append(HERE)  # after site-packages; this judge's frozen_problem.py overrides any candidate copy
    import frozen_problem as fp

    shape = json.dumps([HEIGHT, 2 ** (HEIGHT + 1) - 1, BATCH, ROUNDS])
    try:
        proc = subprocess.run([sys.executable, "-P", __file__, "--build"], input=shape, capture_output=True,
                              text=True, timeout=300)
    except subprocess.TimeoutExpired:
        fail("build_kernel did not finish within 300s")
    if proc.returncode:
        fail(f"build_kernel crashed:\n{proc.stderr[-1500:]}")
    try:
        program = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        fail("could not read the instruction bundles that build_kernel produced")

    cycles = []
    for seed in SEEDS:
        random.seed(seed)
        forest = fp.Tree.generate(HEIGHT)
        inp = fp.Input.generate(forest, BATCH, ROUNDS)
        mem = fp.build_mem_image(forest, inp)
        machine = fp.Machine(mem, program, fp.DebugInfo(scratch_map={}), n_cores=fp.N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        try:
            machine.run()
        except Exception as e:
            fail(f"the program failed on the simulator: {type(e).__name__}: {e}")
        for ref in fp.reference_kernel2(mem):
            pass
        start, n = ref[6], len(inp.values)
        if machine.mem[start:start + n] != ref[start:start + n]:
            fail(f"wrong output values on random input {seed}")
        cycles.append(machine.cycle)
    worst = max(cycles)
    print(json.dumps({"score": worst, "feedback": f"correct on {len(SEEDS)} random inputs; {worst:,} cycles, "
                                                  f"{BASELINE / worst:.2f}x faster than the starting kernel"}))


def worker():
    height, n_nodes, batch, rounds = json.load(sys.stdin)
    sys.path.append(HERE)
    from perf_takehome import KernelBuilder

    kb = KernelBuilder()
    kb.build_kernel(height, n_nodes, batch, rounds)
    print("\n" + json.dumps(kb.instrs))


if __name__ == "__main__":
    worker() if sys.argv[1:] == ["--build"] else main()
