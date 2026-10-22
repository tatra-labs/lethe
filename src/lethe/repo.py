"""The search state lives in git: every candidate is a commit, every agent a worktree.

Each round of a run gets a brand-new repository seeded with the best candidate
so far, so forgetting is literal: the new repository has no history to read.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path

_CONFIG = [
    "-c", "user.name=lethe", "-c", "user.email=lethe@localhost",
    "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", "-c", "gc.auto=0",
]


class Repo:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()  # agents share one repository per round

    def git(self, *args: str, cwd: Path | None = None) -> str:
        with self._lock:
            proc = subprocess.run(
                ["git", *_CONFIG, *args], cwd=cwd or self.path,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        if proc.returncode:
            raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout.strip()

    @classmethod
    def create(cls, path: Path, seed: Path) -> tuple[Repo, str]:
        """A new repository whose only commit holds the files in `seed`."""
        path.mkdir(parents=True)
        shutil.copytree(seed, path, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".context"))
        repo = cls(path)
        repo.git("init", "-q")
        (path / ".git" / "info").mkdir(exist_ok=True)
        (path / ".git" / "info" / "exclude").write_text(".context/\n__pycache__/\n", encoding="utf-8")
        repo.git("add", "-A")
        repo.git("commit", "-q", "--allow-empty", "-m", "starting candidate")
        return repo, repo.git("rev-parse", "HEAD")

    def worktree(self, path: Path, branch: str, commit: str) -> Path:
        self.git("worktree", "add", "-q", "-B", branch, str(path), commit)
        return path

    def commit(self, worktree: Path, message: str) -> str | None:
        """Commit everything in the worktree, or return None if nothing changed."""
        self.git("add", "-A", cwd=worktree)
        if not self.git("status", "--porcelain", cwd=worktree):
            return None
        self.git("commit", "-q", "-m", message, cwd=worktree)
        sha = self.git("rev-parse", "HEAD", cwd=worktree)
        self.git("update-ref", f"refs/candidates/{sha}", sha)  # rejected candidates stay showable
        return sha

    def reset(self, worktree: Path, commit: str) -> None:
        """Make the worktree exactly `commit` again."""
        self.git("reset", "-q", "--hard", commit, cwd=worktree)
        self.git("clean", "-qfdx", "-e", ".context", cwd=worktree)

    def diff(self, a: str, b: str) -> str:
        return self.git("diff", a, b)

    def lines_changed(self, a: str, b: str) -> int:
        total = 0
        for line in self.git("diff", "--numstat", a, b).splitlines():
            added, deleted, _ = line.split("\t", 2)
            if added != "-":  # binary files report "-"
                total += int(added) + int(deleted)
        return total

    def export(self, commit: str, dest: Path) -> None:
        """Replace the contents of `dest` with the files of `commit`."""
        with self._lock:
            data = subprocess.run(
                ["git", *_CONFIG, "archive", "--format=tar", commit],
                cwd=self.path, capture_output=True, check=True,
            ).stdout
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            try:
                tar.extractall(dest, filter="data")
            except TypeError:  # Python < 3.11.4 has no extraction filters
                tar.extractall(dest)
