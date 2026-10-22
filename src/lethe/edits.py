"""SEARCH/REPLACE edit blocks: how a text-only agent changes files.

    solution.py
    <<<<<<< SEARCH
    exact lines to find
    =======
    lines to put there
    >>>>>>> REPLACE

An empty SEARCH section replaces the whole file. A reply with no blocks at all
but one large code fence is taken as a rewrite of the main file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BLOCK = re.compile(
    r"^[ \t]*<{5,9} ?SEARCH[^\n]*\n(?P<search>.*?)^[ \t]*={5,9}[ \t]*\n(?P<replace>.*?)^[ \t]*>{5,9} ?REPLACE[^\n]*$",
    re.DOTALL | re.MULTILINE,
)
FENCE = re.compile(r"^```[^\n]*\n(?P<body>.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Edit:
    path: str | None
    search: str
    replace: str


def strip_thinking(text: str) -> str:
    text = THINK.sub("", text)
    return text.split("</think>", 1)[1] if "</think>" in text else text


def parse(text: str, files: list[str]) -> list[Edit]:
    """Extract edit blocks, attributing each to the file named on the line above it."""
    edits = []
    for m in BLOCK.finditer(text):
        path = _path_before(text, m.start(), files)
        edits.append(Edit(path, m.group("search"), m.group("replace")))
    return edits


def _path_before(text: str, pos: int, files: list[str]) -> str | None:
    for line in reversed(text[:pos].splitlines()[-4:]):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        name = re.sub(r"^(#+|file:|path:)\s*", "", line, flags=re.I).strip("`*: ").replace("\\", "/")
        for f in files:
            if name == f or name.endswith("/" + f) or Path(name).name == Path(f).name:
                return f
        # A new file: accept a plain relative file name, never one that escapes the workspace.
        if re.fullmatch(r"[\w.-]+(/[\w.-]+)*\.\w+", name) and ".." not in name.split("/"):
            return name.removeprefix("./")
        return None
    return None


def apply(edits: list[Edit], workspace: Path, files: list[str]) -> tuple[int, list[str]]:
    """Apply edits in order. Returns (number applied, error messages)."""
    applied, errors = 0, []
    for e in edits:
        path = e.path or (files[0] if len(files) == 1 else None)
        if path is None:
            errors.append("an edit block did not name which file it changes")
            continue
        target = workspace / path
        source = target.read_text(encoding="utf-8") if target.exists() else ""
        if not e.search.strip():
            new = e.replace
        else:
            new = _replace(source, e.search, e.replace)
            if new is None:
                first = e.search.strip().splitlines()[0][:80]
                errors.append(f"SEARCH text not found in {path} (starting with: {first!r})")
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new, encoding="utf-8", newline="\n")
        applied += 1
    return applied, errors


def _replace(source: str, search: str, replace: str) -> str | None:
    if search in source:
        return source.replace(search, replace, 1)
    # Retry ignoring trailing whitespace, which models often get wrong.
    src, pat = source.splitlines(keepends=True), search.splitlines()
    want = [line.rstrip() for line in pat]
    n = len(pat)
    for i in range(len(src) - n + 1):
        if [line.rstrip() for line in src[i:i + n]] == want:
            tail = "" if replace.endswith("\n") or not replace else "\n"
            return "".join(src[:i]) + replace + tail + "".join(src[i + n:])
    return None


def rewrite(text: str, main: Path, min_fraction: float = 0.3) -> bool:
    """Fallback: treat the largest code fence as the new main file, if it is plausibly whole."""
    bodies = [m.group("body") for m in FENCE.finditer(text)]
    if not bodies or not main.exists():
        return False
    body = max(bodies, key=len)
    if len(body) < min_fraction * len(main.read_text(encoding="utf-8")):
        return False
    main.write_text(body, encoding="utf-8", newline="\n")
    return True
