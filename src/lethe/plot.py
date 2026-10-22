"""Charts: the best score so far against the budget spent, with every forget marked, and
how big each edit is at each step of a session (sessions narrow; forgetting resets them)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from .harness import PRESETS, Strategy, read_events

COLORS = {"single": "#8A8F99", "parallel": "#E0A100", "team": "#3B7A57", "forget": "#1F5FBF"}
LABELS = {"single": "1 agent", "parallel": "4 parallel agents", "team": "team of 4",
          "forget": "team of 4 that forgets"}


def kind(run_dir: Path) -> str:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    strategy = Strategy(cfg["agents"], cfg["team"], cfg["rounds"])
    return next((name for name, s in PRESETS.items() if s == strategy),
                f"{strategy.agents} agents{', team' if strategy.team else ''}, {strategy.rounds} rounds")


def curve(events: list[dict], x: str = "proposals") -> tuple[list[float], list[float], list[float]]:
    """Points of the best-so-far step curve, and the x positions of forgets."""
    xs, ys, forgets, spent = [], [], [], 0.0
    for e in events:
        if e.get("event") == "start":
            xs, ys = [0.0], [e["score"]]
        elif e.get("event") == "forget":
            forgets.append(spent)
        elif "status" in e:
            spent += {"proposals": 1, "tokens": e.get("tokens_in", 0) + e.get("tokens_out", 0),
                      "cost": e.get("cost", 0.0)}[x]
            xs.append(spent)
            ys.append(e["best"])
    return xs, ys, forgets


def edit_sizes(events: list[dict]) -> dict[int, list[int]]:
    """Lines changed by each scored proposal, keyed by its step within the session."""
    sizes = defaultdict(list)
    for e in events:
        if "lines" in e:
            sizes[e["step"]].append(e["lines"])
    return sizes


def plot(runs: list[str | Path], out: str | Path, x: str = "proposals", edits: bool = False,
         title: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [Path(r) for r in runs]
    fig, axes = plt.subplots(1, 2 if edits else 1, figsize=(12 if edits else 7.5, 4.6), squeeze=False)
    ax = axes[0][0]
    for run in runs:
        name, events = kind(run), read_events(run)
        color = COLORS.get(name, None)
        xs, ys, forgets = curve(events, x)
        points = [(a, b) for a, b in zip(xs, ys) if b is not None]
        if not points:
            continue
        ax.step(*zip(*points), where="post", color=color, lw=2.2, label=LABELS.get(name, name))
        for f in forgets:
            ax.axvline(f, color=color or "gray", ls=":", lw=1.2, alpha=0.8)
    ax.set_xlabel({"proposals": "proposals (all agents)", "tokens": "tokens", "cost": "dollars"}[x])
    ax.set_ylabel("best score so far")
    ax.set_title(title or "Best score so far; dotted lines are forgets", fontsize=11)
    ax.legend(frameon=False)
    _style(ax)

    if edits:
        ax = axes[0][1]
        for run in runs:
            name, sizes = kind(run), edit_sizes(read_events(run))
            steps = sorted(sizes)
            ax.plot([s + 1 for s in steps], [statistics.median(sizes[s]) for s in steps], "o-",
                    color=COLORS.get(name), lw=1.8, ms=3.5, label=LABELS.get(name, name))
        ax.set_xlabel("step within a session")
        ax.set_ylabel("median lines changed")
        ax.set_title("Edits shrink as a session grows", fontsize=11)
        ax.set_yscale("symlog", linthresh=10)
        ax.legend(frameon=False)
        _style(ax)

    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
