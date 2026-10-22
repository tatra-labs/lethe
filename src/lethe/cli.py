"""lethe run | compare | plot"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .agents import ClaudeCode, LLMAgent
from .harness import PRESETS, Run, Strategy
from .task import Task


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="lethe", description="Agent teams that forget.")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one strategy on a task")
    run.add_argument("task", help="task directory, e.g. tasks/tsp")
    run.add_argument("-s", "--strategy", default="forget", choices=PRESETS,
                     help="single | parallel | team | forget (default)")
    run.add_argument("--agents", type=int, help="override the number of agents")
    run.add_argument("--rounds", type=int, help="override the number of rounds (forgets + 1)")
    run.add_argument("--team", action=argparse.BooleanOptionalAction, default=None,
                     help="override whether agents see each other's candidates")
    _common(run)

    compare = sub.add_parser("compare", help="run several strategies with the same budget, then plot them")
    compare.add_argument("task")
    compare.add_argument("--strategies", default="single,parallel,forget")
    _common(compare)

    chart = sub.add_parser("plot", help="plot one or more runs")
    chart.add_argument("runs", nargs="+")
    chart.add_argument("-o", "--out", default="lethe.png")
    chart.add_argument("--x", default="proposals", choices=["proposals", "tokens", "cost"])
    chart.add_argument("--edits", action="store_true", help="add the edit-size panel")
    chart.add_argument("--title")

    args = p.parse_args(argv)
    if args.command == "plot":
        from .plot import plot

        print(plot(args.runs, args.out, args.x, args.edits, args.title))
        return

    task = Task.load(args.task)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.command == "run":
        base = PRESETS[args.strategy]
        strategy = Strategy(
            agents=args.agents or base.agents,
            team=base.team if args.team is None else args.team,
            rounds=args.rounds or base.rounds,
        )
        out = Path(args.out or f"runs/{task.name}-{args.strategy}-{stamp}")
        Run(task, strategy, _agent_factory(args, task), args.budget, out, args.judges).run()
        return

    out = Path(args.out or f"runs/{task.name}-compare-{stamp}")
    names = [s.strip() for s in args.strategies.split(",")]
    for name in names:
        print(f"\n=== {name}: {PRESETS[name]}")
        Run(task, PRESETS[name], _agent_factory(args, task), args.budget, out / name, args.judges).run()
    from .plot import plot

    print(plot([out / n for n in names], out / "compare.png", edits=True, title=f"{task.name}, budget {args.budget}"))


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-b", "--budget", type=int, default=64, help="total proposals across all agents (default 64)")
    p.add_argument("-o", "--out", help="run directory")
    p.add_argument("--judges", type=int, default=1, help="judges allowed to run at once (default 1)")
    g = p.add_argument_group("agent")
    g.add_argument("--agent", default="llm", choices=["llm", "claude"],
                   help="llm: any OpenAI-compatible endpoint (default); claude: the Claude Code CLI")
    g.add_argument("-m", "--model", help="model name (claude default: sonnet)")
    g.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1"))
    g.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--max-context", type=int, default=32_000, help="session length in tokens before trimming")
    g.add_argument("--max-tokens", type=int, default=8192)
    g.add_argument("--no-think", action="store_true", help="ask Qwen-style models not to think")
    g.add_argument("--extra", default="{}", help="JSON merged into every request body")
    g.add_argument("--price", default="0,0", help="USD per million input,output tokens, for cost tracking")
    g.add_argument("--claude-args", default="", help="extra arguments for the claude CLI")


def _agent_factory(args: argparse.Namespace, task: Task):
    if args.agent == "claude":
        return lambda i: ClaudeCode(model=args.model or "sonnet", extra_args=tuple(args.claude_args.split()))
    if not args.model:
        raise SystemExit("--model is required for --agent llm")
    extra = json.loads(args.extra)
    if args.no_think:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    price = tuple(float(v) for v in args.price.split(","))
    return lambda i: LLMAgent(
        model=args.model, base_url=args.base_url, api_key=os.environ.get(args.api_key_env),
        files=task.files, max_context=args.max_context, temperature=args.temperature,
        max_tokens=args.max_tokens, price=price, extra=extra,
    )
