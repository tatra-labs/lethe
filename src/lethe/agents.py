"""Agents edit a workspace, one message per turn, and keep a session until told to forget.

    LLMAgent     any OpenAI-compatible chat endpoint (OpenAI, OpenRouter, DeepSeek, vLLM,
                 llama.cpp, Ollama, ...); sees the workspace pasted in, replies with edit blocks
    ClaudeCode   the Claude Code CLI, which reads `.context/` and edits files with its own tools

Anything with `uses_tools`, `turn()` and `forget()` can be an agent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import edits

SNAPSHOT_FILE_LIMIT = 60_000  # characters of any one file pasted into a text agent's prompt
PEER_DIFF_LINES = 150


@dataclass
class Turn:
    summary: str = ""  # what the agent says it changed
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    error: str = ""  # anything that went wrong, shown to the agent on its next turn


class Agent(Protocol):
    uses_tools: bool

    def turn(self, message: str, workspace: Path) -> Turn: ...

    def forget(self) -> None: ...


SYSTEM = """You are an autoresearch agent. Over many turns you improve a candidate solution to a task.
Each turn you see the judge's verdict on your current candidate and reply with edits; you cannot run code.

Change files with SEARCH/REPLACE blocks, each preceded by the file's path on its own line:

solution.py
<<<<<<< SEARCH
lines copied exactly from the current file
=======
the lines that replace them
>>>>>>> REPLACE

The SEARCH part must match the current file exactly, including indentation: keep it short but unique.
Use one block per change. A block with an empty SEARCH part replaces the whole file."""


class LLMAgent:
    """A text-only agent over an OpenAI-compatible `/chat/completions` endpoint."""

    uses_tools = False

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        api_key: str | None = None,
        files: list[str] | None = None,
        max_context: int = 32_000,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        price: tuple[float, float] = (0.0, 0.0),
        extra: dict | None = None,
        timeout: float = 1800,
    ):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.files = list(files or [])
        self.max_context, self.temperature, self.max_tokens = max_context, temperature, max_tokens
        self.price, self.extra, self.timeout = price, dict(extra or {}), timeout
        self.history: list[dict] = []

    def forget(self) -> None:
        self.history = []

    def turn(self, message: str, workspace: Path) -> Turn:
        context, files = render_context(workspace), render_files(workspace, self.files)
        user = f"{message}\n\n{context}\n\n{files}"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, *self._window(len(user)),
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra,
        }
        reply = chat(self.base_url, self.api_key, payload, self.timeout)
        text = edits.strip_thinking(reply["choices"][0]["message"].get("content") or "")
        # The session keeps the exchange, minus the bulky file snapshot.
        self.history += [{"role": "user", "content": f"{message}\n\n{context}"}, {"role": "assistant", "content": text}]

        blocks = edits.parse(text, self.files)
        applied, errors = edits.apply(blocks, workspace, self.files)
        if not blocks and self.files and edits.rewrite(text, workspace / self.files[0]):
            applied = 1
        if not applied and not errors:
            errors = ["the reply contained no edit blocks"]

        usage = reply.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        return Turn(
            summary=_summary(text), tokens_in=tin, tokens_out=tout,
            cost=(tin * self.price[0] + tout * self.price[1]) / 1e6, error="; ".join(errors),
        )

    def _window(self, reserve: int) -> list[dict]:
        """The session, trimmed from the oldest exchanges to fit the context; the first stays."""
        budget = self.max_context * 3.5 - reserve - len(SYSTEM)  # ~3.5 characters per token
        head, rest = self.history[:2], self.history[2:]
        budget -= sum(len(m["content"]) for m in head)
        kept: list[dict] = []
        for i in range(len(rest) - 2, -1, -2):
            size = len(rest[i]["content"]) + len(rest[i + 1]["content"])
            if size > budget:
                break
            budget -= size
            kept = rest[i:i + 2] + kept
        return head + kept


class ClaudeCode:
    """Claude Code in headless mode; `--resume` continues the session, forgetting drops it.

    Only file tools and read-only git are loaded, and no MCP servers, which keeps every call
    small (about 13k tokens of fixed context) and runs the same on every machine.
    """

    uses_tools = True
    TOOLS = "Read,Edit,Write,Glob,Grep,Bash(git show *),Bash(git diff *),Bash(git log *)"

    def __init__(self, model: str = "sonnet", tools: str = TOOLS, extra_args: tuple[str, ...] = (),
                 timeout: float = 3600, executable: str = "claude"):
        self.model, self.tools, self.extra_args = model, tools, tuple(extra_args)
        self.timeout, self.executable = timeout, executable
        self.session: str | None = None

    def forget(self) -> None:
        self.session = None

    def turn(self, message: str, workspace: Path) -> Turn:
        cmd = [shutil.which(self.executable) or self.executable, "-p", "--output-format", "json",
               "--permission-mode", "acceptEdits", "--tools", "Read,Edit,Write,Glob,Grep,Bash",
               "--allowedTools", self.tools, "--strict-mcp-config", "--model", self.model, *self.extra_args]
        if self.session:
            cmd += ["--resume", self.session]
        proc = subprocess.run(cmd, input=message, cwd=workspace, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=self.timeout)
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return Turn(error=f"claude exited with code {proc.returncode}: {(proc.stderr or proc.stdout)[-800:]}")
        self.session = data.get("session_id") or self.session
        usage = data.get("usage") or {}
        tin = sum(int(usage.get(k) or 0) for k in
                  ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        return Turn(
            summary=_summary(data.get("result") or ""), tokens_in=tin,
            tokens_out=int(usage.get("output_tokens") or 0), cost=float(data.get("total_cost_usd") or 0.0),
            error=str(data.get("result", ""))[:800] if data.get("is_error") else "",
        )


def chat(base_url: str, api_key: str | None, payload: dict, timeout: float, retries: int = 4) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            request = urllib.request.Request(f"{base_url}/chat/completions", data=body, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if e.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(f"HTTP {e.code} from {base_url}: {e.read()[:500]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(5 * 2 ** attempt)
    raise AssertionError("unreachable")


def render_context(workspace: Path) -> str:
    ctx = workspace / ".context"
    parts = ["# .context"]
    for name in ("feedback.md", "history.md"):
        if (ctx / name).exists():
            parts.append((ctx / name).read_text(encoding="utf-8").strip())
    for peer in sorted((ctx / "peers").glob("branch*")):
        diff_file = peer / "best.diff"
        diff = diff_file.read_text(encoding="utf-8").splitlines() if diff_file.exists() else []
        cut = "\n... (diff truncated)" if len(diff) > PEER_DIFF_LINES else ""
        body = "\n".join(diff[:PEER_DIFF_LINES]) + cut if diff else "(same as your current candidate)"
        parts.append(f"## Peer {peer.name}: best candidate as a diff against yours\n```diff\n{body}\n```")
    return "\n\n".join(parts)


def render_files(workspace: Path, files: list[str]) -> str:
    listed = [workspace / f for f in files]
    others = sorted(p for p in workspace.rglob("*") if p.is_file() and p not in listed
                    and not {".git", ".context", "__pycache__"} & set(p.relative_to(workspace).parts))
    parts = ["# Workspace files"]
    for path in [p for p in listed if p.exists()] + others:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if len(text) > SNAPSHOT_FILE_LIMIT:
            text = text[:SNAPSHOT_FILE_LIMIT] + "\n... (file truncated)"
        parts.append(f"{path.relative_to(workspace).as_posix()}\n```\n{text}\n```")
    return "\n\n".join(parts)


def _summary(text: str) -> str:
    """The last line of prose in a reply (not a file path), as a one-line summary of the change."""
    prose = [line.strip().strip("`*#: ") for line in edits.BLOCK.sub("", edits.FENCE.sub("", text)).splitlines()]
    prose = [line for line in prose if line and not line.startswith(("<<<<<<<", "=======", ">>>>>>>"))
             and not re.fullmatch(r"[\w./\\-]+\.\w+", line)]
    return prose[-1][:200] if prose else ""
