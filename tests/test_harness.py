import json
import subprocess
from pathlib import Path

import pytest

from lethe import Run, Strategy, Task, Turn, read_events

TOY = Path(__file__).parent / "toy"


class Adder:
    """Adds `step` to X every turn and remembers what it was told."""

    uses_tools = True

    def __init__(self, step=1, write=None):
        self.step, self.write = step, write
        self.messages, self.forgets = [], 0

    def turn(self, message, workspace):
        self.messages.append(message)
        path = workspace / "value.py"
        x = int(path.read_text().split("=")[1])
        path.write_text(self.write if self.write is not None else f"X = {x + self.step}\n", newline="\n")
        return Turn(summary=f"X += {self.step}", tokens_in=10, tokens_out=5)

    def forget(self):
        self.forgets += 1


def run(tmp_path, strategy, budget, **agent):
    agents = []

    def make(i):
        agents.append(Adder(**agent))
        return agents[-1]

    best = Run(Task.load(TOY), strategy, make, budget, tmp_path / "run", log=lambda _: None).run()
    return best, agents, read_events(tmp_path / "run")


def proposals(events):
    return [e for e in events if "status" in e]


def test_single_agent_climbs(tmp_path):
    best, (agent,), events = run(tmp_path, Strategy(1, False, 1), budget=5)
    assert best.score == 5
    assert [e["status"] for e in proposals(events)] == ["accepted"] * 5
    assert (tmp_path / "run" / "best" / "value.py").read_text() == "X = 5\n"
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["best"] == 5
    assert agent.forgets == 0
    assert "# Task" in agent.messages[0] and all("# Continue" in m for m in agent.messages[1:])


def test_forgetting_restarts_every_agent_from_the_best(tmp_path):
    best, agents, events = run(tmp_path, Strategy(2, True, 3), budget=12)
    # Each round, each agent adds 1 twice to the best candidate it restarted from.
    assert best.score == 6
    assert [e["best"] for e in events if e.get("event") == "forget"] == [2, 4]
    assert all(a.forgets == 2 for a in agents)
    # Every round opens a fresh session with the full first-turn message.
    assert sum("# Task" in m for m in agents[0].messages) == 3
    # And a fresh repository: nothing of the earlier rounds is left to read.
    repo = tmp_path / "run" / "round2" / "repo"
    root = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.split()
    assert len(root) == 1
    assert subprocess.run(["git", "show", f"{root[0]}:value.py"], cwd=repo, capture_output=True,
                          text=True).stdout == "X = 4\n"


def test_team_sees_peers_but_parallel_agents_do_not(tmp_path):
    run(tmp_path / "team", Strategy(2, True, 1), budget=4)
    run(tmp_path / "parallel", Strategy(2, False, 1), budget=4)
    team_ctx = tmp_path / "team" / "run" / "round0" / "agent0" / ".context"
    assert "Branch 1's attempts" in (team_ctx / "peers" / "branch1" / "pool.md").read_text()
    assert not (tmp_path / "parallel" / "run" / "round0" / "agent0" / ".context" / "peers").exists()


def test_worse_candidates_are_rejected_and_reset(tmp_path):
    best, (agent,), events = run(tmp_path, Strategy(1, False, 1), budget=3, step=-1)
    assert best.score == 0
    assert [e["status"] for e in proposals(events)] == ["rejected"] * 3
    assert (tmp_path / "run" / "round0" / "agent0" / "value.py").read_text() == "X = 0\n"
    assert "not better" in (tmp_path / "run" / "round0" / "agent0" / ".context" / "feedback.md").read_text()


def test_invalid_and_empty_turns(tmp_path):
    _, _, events = run(tmp_path / "bad", Strategy(1, False, 1), budget=2, write="X = (\n")
    assert [e["status"] for e in proposals(events)] == ["invalid"] * 2
    _, _, events = run(tmp_path / "idle", Strategy(1, False, 1), budget=2, write="X = 0\n")
    assert [e["status"] for e in proposals(events)] == ["empty"] * 2


def test_runs_plot_with_their_forgets(tmp_path):
    from lethe.plot import curve, plot

    _, _, events = run(tmp_path, Strategy(2, True, 2), budget=4)
    xs, ys, forgets = curve(events)
    assert xs == [0, 1, 2, 3, 4] and ys == [0, 1, 1, 2, 2] and forgets == [2]
    assert plot([tmp_path / "run"], tmp_path / "chart.png", edits=True).stat().st_size > 0


def test_budget_must_cover_the_rounds(tmp_path):
    with pytest.raises(ValueError):
        Run(Task.load(TOY), Strategy(4, True, 4), lambda i: Adder(), 8, tmp_path / "run")
