"""The messages a session receives, after the paper's appendix. They describe the task and
the loop, and deliberately give no strategy for the search itself."""

from __future__ import annotations

FIRST = """# Task

{task}

# Feedbacks and Updates

{status}

# Instructions

{instructions}"""

LATER = """# Feedbacks and Updates

{status}

# Continue

{reminder}"""

# For agents with file tools (e.g. Claude Code), which read `.context/` themselves.
TOOL_INSTRUCTIONS = """`.context/` holds what was prepared for this turn: the candidate's feedback, its
lineage, other candidates, or the whole run. Read it before changing anything.
Every candidate is a commit in this repository: `git show <id>:<file>` shows any
of them, `git diff <a> <b>` compares two.

Improve the candidate, then end the session and say what you changed. Do not
evaluate the change yourself: another process scores your candidate after you
stop and invokes you again with the result, so a simulator or judge built here
only spends this session. Do not commit; the harness commits your working tree,
scores it, and continues from it in a later turn. Work inside this directory
only, and there is no need for background tasks. Anything you leave here (tests,
scripts) is committed with your change and is there next turn.

Once your candidate is in place, end the session. Another process evaluates it
and invokes you again."""

TOOL_REMINDER = """The judge's feedback and the run's state are in `.context/` as before;
`.context/TASK.md` holds the full instructions. Improve the candidate, then end the
session. Another process scores it and invokes you again."""

# For text-only agents, which get the workspace and `.context/` pasted below the message.
TEXT_INSTRUCTIONS = """Below are the files of your workspace and `.context/`, which holds what was
prepared for this turn: the judge's feedback on your current candidate, your earlier
attempts and, in a team, your peers' best candidates as diffs against yours.

Improve the candidate: reply with edit blocks, then one line saying what you changed.
Do not evaluate the change yourself: another process applies your edits, scores the
candidate and invokes you again with the result."""

TEXT_REMINDER = """The judge's feedback and the run's state are below as before. Improve the
candidate with edit blocks, then stop. Another process scores it and invokes you again."""


def fmt(score: float | None) -> str:
    if score is None:
        return "invalid"
    if float(score).is_integer() or abs(score) >= 1e5:
        return f"{score:,.0f}"
    return f"{score:.6g}"


def status(round: int, current: tuple[str, float | None], peers: dict[int, tuple[str, float | None]]) -> str:
    lines = [f"Round {round}.", f"Your current candidate is `{current[0]}`, mean score {fmt(current[1])}."]
    for j, (cid, score) in sorted(peers.items()):
        lines.append(f"Branch {j}: best `{cid}` at {fmt(score)}; its pool is under `.context/peers/branch{j}/`.")
    return "\n".join(lines)


def message(first: bool, tools: bool, task: str, status_text: str) -> str:
    if first:
        instructions = TOOL_INSTRUCTIONS if tools else TEXT_INSTRUCTIONS
        return FIRST.format(task=task, status=status_text, instructions=instructions)
    return LATER.format(status=status_text, reminder=TOOL_REMINDER if tools else TEXT_REMINDER)
