from pathlib import Path

from lethe import Task
from lethe.task import parse_verdict

TOY = Path(__file__).parent / "toy"


def test_toy_task_scores_its_candidate():
    task = Task.load(TOY)
    assert task.evaluate(task.candidate_dir).score == 0
    assert task.better(1, 0) and not task.better(None, 0) and task.better(0, None)


def test_verdicts_come_from_the_last_json_line():
    assert parse_verdict(0, 'noise\n{"score": 2.5, "feedback": "ok"}\n', "").score == 2.5
    assert parse_verdict(0, '{"score": null, "feedback": "overlap"}', "").feedback == "overlap"
    bad = parse_verdict(1, "", "Traceback ...")
    assert bad.score is None and "Traceback" in bad.feedback
    # Agents will try anything: absurd scores are invalid, not crashes.
    assert parse_verdict(0, '{"score": ' + "9" * 400 + "}", "").score is None
    assert parse_verdict(0, '{"score": NaN}', "").score is None
    assert parse_verdict(0, '{"score": "high"}', "").score is None


def test_minimizing_tasks_prefer_lower_scores(tmp_path):
    task = Task.load(TOY)
    task.direction = "min"
    assert task.better(1, 2) and not task.better(2, 1)
