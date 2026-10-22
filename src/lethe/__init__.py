"""lethe: agent teams that forget.

    from lethe import Task, Run, Strategy, LLMAgent

    task = Task.load("tasks/tsp")
    Run(task, Strategy.preset("forget"), lambda i: LLMAgent("gpt-5-mini", "https://api.openai.com/v1", key,
        files=task.files), budget=64, out="runs/tsp").run()
"""

from .agents import Agent, ClaudeCode, LLMAgent, Turn
from .harness import PRESETS, Candidate, Run, Strategy, read_events
from .task import Task, Verdict

__all__ = ["Agent", "Candidate", "ClaudeCode", "LLMAgent", "PRESETS", "Run", "Strategy", "Task", "Turn",
           "Verdict", "read_events"]
__version__ = "0.1.0"
