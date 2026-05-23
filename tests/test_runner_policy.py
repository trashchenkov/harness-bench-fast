from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.errors import GraphRecursionError

from harness_bench import __main__ as bench_main
from harness_bench.core import VerifyResult
from harness_bench.runner import TaskRun, run_task


class _FakeAgent:
    def invoke(self, _payload: object) -> None:
        raise GraphRecursionError("Recursion limit of 3 reached without hitting a stop condition")


class _FakeTask:
    id = "task_fake_recursion"
    name = "Fake recursion task"
    prompt = "Do something that loops"

    def setup(self, workspace: Path) -> None:
        (workspace / "input.txt").write_text("fixture", encoding="utf-8")

    def verify(self, _workspace: Path) -> VerifyResult:
        raise AssertionError("verify should not run after agent recursion failure")


def test_graph_recursion_limit_is_task_failure_without_traceback(monkeypatch) -> None:
    monkeypatch.setattr("harness_bench.runner.build_agent", lambda *_args, **_kwargs: _FakeAgent())

    result = run_task(_FakeTask(), recursion_limit=3)

    assert result.passed is False
    assert result.message == "graph recursion limit reached after 3 steps"
    assert result.error is None


def test_strict_run_returns_nonzero_on_task_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        bench_main,
        "run_all",
        lambda **_kwargs: [
            TaskRun("task_fake", False, "expected verifier failure", 0.01),
        ],
    )
    monkeypatch.setattr(bench_main, "summarize", lambda _results: None)

    assert bench_main.main(["run", "--task", "task_fake"]) == 1


def test_allow_task_failures_returns_zero_when_harness_completed(monkeypatch) -> None:
    monkeypatch.setattr(
        bench_main,
        "run_all",
        lambda **_kwargs: [
            TaskRun("task_fake", False, "expected verifier failure", 0.01),
        ],
    )
    monkeypatch.setattr(bench_main, "summarize", lambda _results: None)

    assert bench_main.main(["run", "--task", "task_fake", "--allow-task-failures"]) == 0


@pytest.mark.parametrize(
    ("module_name", "run_all_name", "run_task_name", "extra_kwargs"),
    [
        ("harness_bench.runner", "run_all", "run_task", {"recursion_limit": 3}),
        ("harness_bench.runner_pure", "run_all", "run_task", {"recursion_limit": 3}),
        (
            "harness_bench.runner_openrouter",
            "run_all",
            "run_task",
            {"model_name": "test-model", "recursion_limit": 3},
        ),
        (
            "harness_bench.runner_cli",
            "run_all_cli",
            "run_task_cli",
            {"cli_command": "python -c pass", "timeout": 1},
        ),
    ],
)
def test_sequential_progress_output_is_cp1251_safe(
    module_name: str,
    run_all_name: str,
    run_task_name: str,
    extra_kwargs: dict[str, object],
    monkeypatch,
    capsys,
) -> None:
    module = pytest.importorskip(module_name)
    fake_task = SimpleNamespace(id="task_fake", name="Fake task")

    monkeypatch.setattr(module, "_load_env_from_dotenv", lambda: None)
    if hasattr(module, "_ensure_credentials"):
        monkeypatch.setattr(module, "_ensure_credentials", lambda: None)
    if hasattr(module, "_ensure_openrouter_key"):
        monkeypatch.setattr(module, "_ensure_openrouter_key", lambda: None)
    monkeypatch.setattr(module, "get_task", lambda _task_id: fake_task)
    monkeypatch.setattr(
        module,
        run_task_name,
        lambda *_args, **_kwargs: TaskRun("task_fake", True, "ok", 0.01),
    )

    getattr(module, run_all_name)(task_ids=["task_fake"], concurrency=1, **extra_kwargs)

    output = capsys.readouterr().out
    assert "[START] task_fake: Fake task" in output
    output.encode("cp1251")
