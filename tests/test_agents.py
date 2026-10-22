from pathlib import Path

from lethe import LLMAgent, edits
from lethe import agents as agents_module

SOURCE = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"


def test_parse_and_apply_blocks(tmp_path: Path):
    (tmp_path / "a.py").write_text(SOURCE)
    reply = (
        "Make f return 3.\n\na.py\n```python\n<<<<<<< SEARCH\n    return 1\n=======\n    return 3\n"
        ">>>>>>> REPLACE\n```\n\n`helper.py`\n<<<<<<< SEARCH\n=======\nK = 7\n>>>>>>> REPLACE\n"
    )
    blocks = edits.parse(reply, ["a.py"])
    assert [b.path for b in blocks] == ["a.py", "helper.py"]
    assert edits.apply(blocks, tmp_path, ["a.py"]) == (2, [])
    assert (tmp_path / "a.py").read_text() == SOURCE.replace("return 1", "return 3")
    assert (tmp_path / "helper.py").read_text() == "K = 7\n"


def test_trailing_whitespace_is_forgiven_and_misses_are_reported(tmp_path: Path):
    (tmp_path / "a.py").write_text(SOURCE)
    fuzzy = edits.Edit("a.py", "def g():   \n    return 2\n", "def g():\n    return 4\n")
    missing = edits.Edit("a.py", "def h():\n", "")
    applied, errors = edits.apply([fuzzy, missing], tmp_path, ["a.py"])
    assert applied == 1 and "not found" in errors[0]
    assert "return 4" in (tmp_path / "a.py").read_text()


def test_nearly_identical_blocks_still_apply_but_different_ones_do_not(tmp_path: Path):
    (tmp_path / "a.py").write_text(SOURCE)
    retyped = edits.Edit("a.py", "def f():\n    return 1\n\ndef g():\n    return 2\n", "def f():\n    return 7\n")
    assert edits.apply([retyped], tmp_path, ["a.py"]) == (1, [])  # one blank line was dropped
    assert (tmp_path / "a.py").read_text() == "def f():\n    return 7\n"
    other = edits.Edit("a.py", "class Solver:\n    pass\n", "x = 1\n")
    assert edits.apply([other], tmp_path, ["a.py"])[0] == 0


def test_a_misnamed_file_means_the_only_editable_one(tmp_path: Path):
    (tmp_path / "priority.py").write_text(SOURCE)
    reply = "solution.py\n<<<<<<< SEARCH\n    return 2\n=======\n    return 3\n>>>>>>> REPLACE\n"
    assert edits.apply(edits.parse(reply, ["priority.py"]), tmp_path, ["priority.py"]) == (1, [])
    assert "return 3" in (tmp_path / "priority.py").read_text() and not (tmp_path / "solution.py").exists()


def test_paths_cannot_escape_the_workspace():
    reply = "../../evil.py\n<<<<<<< SEARCH\n=======\nx\n>>>>>>> REPLACE\n"
    assert edits.parse(reply, ["a.py"])[0].path is None


def test_thinking_is_stripped_and_a_lone_fence_is_a_rewrite(tmp_path: Path):
    (tmp_path / "a.py").write_text(SOURCE)
    text = edits.strip_thinking("<think>hmm</think>Rewritten:\n```python\n" + SOURCE.replace("1", "9") + "```\n")
    assert "hmm" not in text
    assert edits.rewrite(text, tmp_path / "a.py")
    assert "return 9" in (tmp_path / "a.py").read_text()


def test_llm_agent_edits_and_keeps_a_lean_session(tmp_path: Path, monkeypatch):
    (tmp_path / "a.py").write_text(SOURCE)
    sent = []

    def fake_chat(base_url, api_key, payload, timeout):
        sent.append(payload)
        content = "a.py\n<<<<<<< SEARCH\n    return 2\n=======\n    return 5\n>>>>>>> REPLACE\nMade g return 5."
        usage = {"prompt_tokens": 100, "completion_tokens": 20}
        return {"choices": [{"message": {"content": content}}], "usage": usage}

    monkeypatch.setattr(agents_module, "chat", fake_chat)
    agent = LLMAgent("m", files=["a.py"], price=(1.0, 2.0))
    turn = agent.turn("# Task\n\nimprove", tmp_path)
    assert "return 5" in (tmp_path / "a.py").read_text()
    assert (turn.tokens_in, turn.tokens_out, turn.summary, turn.error) == (100, 20, "Made g return 5.", "")
    assert turn.cost == (100 * 1.0 + 20 * 2.0) / 1e6
    assert "def f()" in sent[0]["messages"][-1]["content"]  # the file snapshot is sent ...
    assert "def f()" not in agent.history[0]["content"]  # ... but not kept in the session
    agent.turn("# Continue", tmp_path)
    assert len(sent[1]["messages"]) == 4  # system, first exchange, new message
    agent.forget()
    assert agent.history == []


def test_session_window_keeps_the_first_exchange():
    agent = LLMAgent("m", max_context=100)
    agent.history = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "ok"}]
    agent.history += [{"role": "user", "content": "x" * 300}, {"role": "assistant", "content": "y" * 300}] * 3
    window = agent._window(reserve=0)
    assert window[0]["content"] == "task" and len(window) < len(agent.history)
