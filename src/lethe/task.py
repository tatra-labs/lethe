"""A task is a directory with a description, a starting candidate, and a judge.

    my_task/
      task.toml     name, files the agent edits, judge command, direction, timeout
      TASK.md       what every agent is told
      candidate/    the starting candidate; it becomes each agent's workspace
      judge/        scoring code, copied next to the candidate only at judging time

The judge prints one JSON line, ``{"score": 1.23, "feedback": "..."}``, as the
last line of its stdout. A missing score, a crash or a timeout makes the
candidate invalid.
"""

from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

FEEDBACK_LIMIT = 4000


@dataclass
class Verdict:
    score: float | None  # None: the candidate is invalid
    feedback: str = ""


@dataclass
class Task:
    root: Path
    name: str
    description: str
    judge: str
    files: list[str] = field(default_factory=list)
    direction: str = "max"
    timeout: float = 300.0
    setup: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> Task:
        root = Path(path).resolve()
        cfg = tomllib.loads((root / "task.toml").read_text(encoding="utf-8"))
        task = cls(
            root=root,
            name=cfg.get("name", root.name),
            description=(root / "TASK.md").read_text(encoding="utf-8").strip(),
            judge=cfg["judge"],
            files=list(cfg.get("files", [])),
            direction=cfg.get("direction", "max"),
            timeout=float(cfg.get("timeout", 300)),
            setup=cfg.get("setup"),
        )
        if task.direction not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min', not {task.direction!r}")
        if not task.candidate_dir.exists() and task.setup:
            subprocess.run(task.setup.replace("{python}", sys.executable), shell=True, cwd=root, check=True)
        if not task.candidate_dir.exists():
            raise FileNotFoundError(f"{task.candidate_dir} does not exist")
        return task

    @property
    def candidate_dir(self) -> Path:
        return self.root / "candidate"

    def better(self, a: float | None, b: float | None) -> bool:
        """True if score `a` is strictly better than `b`; invalid scores lose to everything."""
        if a is None:
            return False
        if b is None:
            return True
        return a > b if self.direction == "max" else a < b

    def evaluate(self, workspace: Path) -> Verdict:
        """Score the candidate in `workspace` inside a throwaway copy, never in place."""
        with tempfile.TemporaryDirectory(prefix="lethe-judge-", ignore_cleanup_errors=True) as tmp:
            sandbox = Path(tmp)
            shutil.copytree(workspace, sandbox, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".context"))
            # Judge files go in last, so a candidate cannot shadow them.
            shutil.copytree(self.root / "judge", sandbox, dirs_exist_ok=True)
            # No shell: on timeout the judge itself is killed, not just a wrapper around it.
            command = [sys.executable if arg == "{python}" else arg for arg in shlex.split(self.judge)]
            try:
                proc = subprocess.run(
                    command, cwd=sandbox, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return Verdict(None, f"judge timed out after {self.timeout:.0f}s")
        return parse_verdict(proc.returncode, proc.stdout, proc.stderr)


def parse_verdict(returncode: int, stdout: str, stderr: str) -> Verdict:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                break
            feedback = str(data.get("feedback", ""))[:FEEDBACK_LIMIT]
            if data.get("score") is None:
                return Verdict(None, feedback)
            try:
                score = float(data["score"])
            except (TypeError, ValueError, OverflowError):
                score = math.nan
            if not math.isfinite(score):
                return Verdict(None, f"the judge reported a score that is not a finite number: "
                                     f"{str(data['score'])[:80]}\n{feedback}")
            return Verdict(score, feedback)
    tail = (stderr.strip() or stdout.strip())[-FEEDBACK_LIMIT:]
    return Verdict(None, f"judge exited with code {returncode} and no score:\n{tail}")
