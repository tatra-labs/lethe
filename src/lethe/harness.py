"""The run loop. Agents propose, the judge scores, and the reducer decides what each agent
sees and remembers. Forgetting splits the budget into rounds: at the end of each round
every session and the whole search state are discarded, and all agents restart from the
single best candidate found so far.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import prompts
from .agents import Agent, Turn
from .repo import Repo
from .task import Task, Verdict

DIFF_LIMIT = 400  # lines of each peer diff written to .context/


@dataclass(frozen=True)
class Strategy:
    agents: int = 4
    team: bool = True  # agents see each other's candidates
    rounds: int = 4  # the run forgets rounds - 1 times

    @staticmethod
    def preset(name: str) -> Strategy:
        return PRESETS[name]


PRESETS = {
    "single": Strategy(agents=1, team=False, rounds=1),  # one agent, one session, whole budget
    "parallel": Strategy(agents=4, team=False, rounds=1),  # four agents that never see each other
    "team": Strategy(agents=4, team=True, rounds=1),  # four agents that share candidates
    "forget": Strategy(agents=4, team=True, rounds=4),  # a team that forgets three times
}


@dataclass
class Candidate:
    id: str
    parent: str | None
    agent: int
    round: int
    step: int
    score: float | None
    feedback: str = ""
    summary: str = ""
    accepted: bool = False

    @property
    def short(self) -> str:
        return self.id[:7]


class Run:
    def __init__(
        self,
        task: Task,
        strategy: Strategy,
        make_agent: Callable[[int], Agent],
        budget: int,
        out: str | Path,
        judges: int = 1,
        log: Callable[[str], None] = print,
    ):
        per_agent = budget // strategy.agents
        if per_agent < strategy.rounds:
            raise ValueError(f"a budget of {budget} proposals is too small for {strategy}")
        self.task, self.strategy, self.make_agent, self.budget = task, strategy, make_agent, budget
        self.out, self.log = Path(out), log
        # Every agent gets the same share of the budget, split evenly over the rounds.
        self.steps = [per_agent // strategy.rounds] * strategy.rounds
        self.steps[-1] += per_agent % strategy.rounds
        self._judges = threading.Semaphore(judges)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.n = 0  # proposals made so far
        self.best: Candidate | None = None

    def run(self) -> Candidate:
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"{self.out} already holds a run")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps({
            "task": self.task.name, "direction": self.task.direction, "budget": self.budget,
            "steps_per_round": self.steps, **asdict(self.strategy),
        }, indent=2), encoding="utf-8")
        self._events = open(self.out / "events.jsonl", "a", encoding="utf-8")
        try:
            agents = [self.make_agent(i) for i in range(self.strategy.agents)]
            seed = self.task.candidate_dir
            for k, steps in enumerate(self.steps):
                if k:
                    self._emit({"event": "forget", "round": k, "n": self.n, "best": self.best.score,
                                "id": self.best.id})
                    self.log(f"-- forget: every session and candidate is dropped; restarting from "
                             f"`{self.best.short}` ({prompts.fmt(self.best.score)})")
                    for agent in agents:
                        agent.forget()
                    seed = self.out / "best"
                self._round(k, seed, agents, steps)
        except KeyboardInterrupt:
            self.log("interrupted")
        finally:
            self._events.close()
        if self.best is not None:
            self._write_summary()
        return self.best

    def _round(self, k: int, seed: Path, agents: list[Agent], steps: int) -> None:
        rdir = self.out / f"round{k}"
        repo, root = Repo.create(rdir / "repo", seed)
        if self.best is None:
            verdict = self.task.evaluate(repo.path)
            start = Candidate(root, None, -1, k, 0, verdict.score, verdict.feedback, "starting candidate", True)
            self._emit({"event": "start", "n": 0, "id": root, "score": start.score, "best": start.score})
            self.log(f"starting candidate scores {prompts.fmt(start.score)}")
            self.best = start
            repo.export(root, self.out / "best")
        else:
            start = Candidate(root, None, -1, k, 0, self.best.score, self.best.feedback, "best so far", True)
        self.repo = repo
        self.workspaces = [repo.worktree(rdir / f"agent{i}", f"agent{i}", root) for i in range(len(agents))]
        self.current = [start] * len(agents)  # each agent's candidate to improve
        self.pools: list[list[Candidate]] = [[] for _ in agents]  # each agent's attempts this round
        self.notes = [""] * len(agents)  # what happened on each agent's last turn
        with ThreadPoolExecutor(len(agents)) as pool:
            futures = [pool.submit(self._agent_loop, i, agents[i], k, steps) for i in range(len(agents))]
            try:
                for f in futures:
                    f.result()
            except KeyboardInterrupt:
                self._stop.set()  # agents finish the turn in flight, then stop
                self.log("stopping after the turns in flight ...")
                raise

    def _agent_loop(self, i: int, agent: Agent, k: int, steps: int) -> None:
        ws = self.workspaces[i]
        for step in range(steps):
            if self._stop.is_set():
                return
            peers = self._write_context(i, ws)
            status = prompts.status(k + 1, (self.current[i].short, self.current[i].score), peers)
            message = prompts.message(step == 0, agent.uses_tools, self.task.description, status)
            t0 = time.time()
            try:
                turn = agent.turn(message, ws)
            except Exception as e:  # a failed call costs the turn, not the run
                turn = Turn(error=f"{type(e).__name__}: {e}")
            seconds = time.time() - t0
            self._record(i, k, step, turn, seconds)

    def _record(self, i: int, k: int, step: int, turn: Turn, seconds: float) -> None:
        ws, parent = self.workspaces[i], self.current[i]
        if turn.reply or turn.error:
            replies = self.out / "replies"
            replies.mkdir(exist_ok=True)
            (replies / f"round{k + 1}-agent{i}-step{step + 1}.md").write_text(
                turn.reply + (f"\n\n<!-- harness: {turn.error} -->\n" if turn.error else ""), encoding="utf-8")
        sha = self.repo.commit(ws, f"agent {i}, round {k + 1}, step {step + 1}: {turn.summary}"[:200])
        event = {"agent": i, "round": k, "step": step, "parent": parent.id, "tokens_in": turn.tokens_in,
                 "tokens_out": turn.tokens_out, "cost": turn.cost, "seconds": round(seconds, 2)}
        if sha is None:
            self.notes[i] = turn.error or "your last turn changed no files, so there was nothing to score"
            self._finish(i, {**event, "status": "empty", "error": turn.error})
            return
        lines = self.repo.lines_changed(parent.id, sha)
        with self._judges:
            t0 = time.time()
            try:
                verdict = self.task.evaluate(ws)
            except Exception as e:  # a broken candidate must never stop the run
                verdict = Verdict(None, f"judging failed: {type(e).__name__}: {e}")
            event["judge_seconds"] = round(time.time() - t0, 2)
        cand = Candidate(sha, parent.id, i, k, step, verdict.score, verdict.feedback, turn.summary)
        cand.accepted = self.task.better(cand.score, parent.score)
        if cand.accepted:
            self.current[i] = cand
            self.notes[i] = turn.error and f"Some edit blocks failed: {turn.error}"
        else:
            self.repo.reset(ws, parent.id)
            verdict_text = "invalid" if cand.score is None else f"scored {prompts.fmt(cand.score)}, not better"
            self.notes[i] = (f"Your last candidate `{cand.short}` {verdict_text}; the workspace was reset to "
                             f"`{parent.short}`.\nJudge feedback on it:\n{cand.feedback or '(none)'}"
                             + (f"\nEdit problems: {turn.error}" if turn.error else ""))
        status = "accepted" if cand.accepted else ("invalid" if cand.score is None else "rejected")
        self._finish(i, {**event, "id": sha, "status": status, "score": cand.score, "lines": lines,
                         "summary": turn.summary}, cand)

    def _finish(self, i: int, event: dict, cand: Candidate | None = None) -> None:
        with self._lock:
            self.n += 1
            if cand is not None:
                self.pools[i].append(cand)
                if self.task.better(cand.score, self.best.score):
                    self.best = cand
                    self.repo.export(cand.id, self.out / "best")
            event = {"n": self.n, **event, "best": self.best.score}
            self._emit(event)
        score = prompts.fmt(event["score"]) if "score" in event else "-"
        self.log(f"r{event['round'] + 1} agent{i} step {event['step'] + 1:>3}  {event['status']:<8} "
                 f"{score:>12}   best {prompts.fmt(self.best.score)}")

    def _emit(self, event: dict) -> None:
        self._events.write(json.dumps({**event, "time": round(time.time(), 3)}) + "\n")
        self._events.flush()

    def _write_context(self, i: int, ws: Path) -> dict[int, tuple[str, float | None]]:
        """Prepare `.context/` for agent i's next turn and return what it sees of its peers."""
        ctx = ws / ".context"
        shutil.rmtree(ctx, ignore_errors=True)
        ctx.mkdir()
        cur = self.current[i]
        (ctx / "TASK.md").write_text(self.task.description + "\n", encoding="utf-8")
        feedback = f"## Your current candidate `{cur.short}`: score {prompts.fmt(cur.score)}\n\n" \
                   f"Judge feedback:\n{cur.feedback or '(none)'}\n"
        if self.notes[i]:
            feedback += f"\n## Your last turn\n\n{self.notes[i]}\n"
        (ctx / "feedback.md").write_text(feedback, encoding="utf-8")
        (ctx / "history.md").write_text(_pool_table("Your attempts this session", self.pools[i]), encoding="utf-8")
        peers = {}
        if self.strategy.team:
            with self._lock:
                snapshot = [(j, self.current[j], list(self.pools[j])) for j in range(len(self.pools)) if j != i]
            for j, best, pool in snapshot:
                pdir = ctx / "peers" / f"branch{j}"
                pdir.mkdir(parents=True)
                (pdir / "pool.md").write_text(_pool_table(f"Branch {j}'s attempts", pool), encoding="utf-8")
                if best.id != cur.id:
                    diff = self.repo.diff(cur.id, best.id).splitlines()
                    (pdir / "best.diff").write_text("\n".join(diff[:DIFF_LIMIT]) + "\n", encoding="utf-8")
                peers[j] = (best.short, best.score)
        return peers

    def _write_summary(self) -> None:
        events = read_events(self.out)
        proposals = [e for e in events if "status" in e]
        summary = {
            "task": self.task.name, "best": self.best.score, "best_id": self.best.id, "proposals": len(proposals),
            "accepted": sum(e["status"] == "accepted" for e in proposals),
            "tokens": sum(e.get("tokens_in", 0) + e.get("tokens_out", 0) for e in proposals),
            "cost": round(sum(e.get("cost", 0.0) for e in proposals), 4),
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.log(f"best {prompts.fmt(self.best.score)} (`{self.best.short}`), files in {self.out / 'best'}")


def _pool_table(title: str, pool: list[Candidate]) -> str:
    if not pool:
        return f"# {title}\n\nNone yet.\n"
    rows = [f"# {title}\n", "| id | parent | score | result | what changed |", "|---|---|---|---|---|"]
    for c in pool:
        result = "accepted" if c.accepted else ("invalid" if c.score is None else "rejected")
        parent = c.parent[:7] if c.parent else "-"
        rows.append(f"| `{c.short}` | `{parent}` | {prompts.fmt(c.score)} | {result} | {c.summary.replace('|', '/')} |")
    return "\n".join(rows) + "\n"


def read_events(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
