"""Tests for live_tests.py itself (no real model): rate-limit / network problems must be reported
as NOT RUN and never recorded as a model FAIL."""
import importlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import core

ROOT = Path(__file__).resolve().parent.parent
RESUME = "Test Candidate\nSKILLS\nPython, SQL, Excel\n\nEXPERIENCE\n- Wrote SQL queries for monthly sales reports.\n" * 2
JD = "Data Analyst. Required: Python, SQL, Tableau. Strong communication skills needed."


@pytest.fixture
def lt(monkeypatch):
    # importing live_tests replaces two functions in core; monkeypatch puts them back after the test
    monkeypatch.setattr(core, "call_groq", core.call_groq)
    monkeypatch.setattr(core, "_friendly_error", core._friendly_error)
    module = importlib.import_module("live_tests")
    return importlib.reload(module)  # re-install the patches for THIS test (the module is cached otherwise)


def test_classify_only_turns_non_passing_infrastructure_problems_into_not_run(lt):
    assert lt.classify(False, "Groq rate limit reached.") == lt.INFRA
    assert lt.classify(None, "Cannot reach Groq.") == lt.INFRA
    assert lt.classify(True, "Groq rate limit reached.") is True   # passed after waiting: still PASS
    assert lt.classify(False, None) is False                        # a genuine model failure stays FAIL
    assert lt.VERDICT[lt.INFRA].startswith("NOT RUN")


def test_a_rate_limited_case_is_detected_as_not_a_model_result(lt, monkeypatch):
    class RateLimitError(Exception):
        pass

    def create(**kwargs):
        raise RateLimitError("Rate limit reached ... Please try again in 3.0s. rate_limit_exceeded")

    monkeypatch.setattr(core, "get_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    lt.TRACE["infra"] = None
    state, report, result, calls = lt.analyze(RESUME, JD)
    assert state is None and "rate limit" in report.lower()
    assert lt.TRACE["infra"] and "rate limit" in lt.TRACE["infra"].lower()
    assert lt.classify(False, lt.TRACE["infra"]) == lt.INFRA


def test_recorder_writes_not_run_instead_of_fail_and_keeps_real_results(lt, monkeypatch, tmp_path):
    shutil.copy(ROOT / "TESTING.md", tmp_path / "TESTING.md")
    monkeypatch.setattr(lt, "ROOT", tmp_path)
    results = [
        {"case": "L1 (run 1)", "pass": True, "summary": {"fit_score": 86}},
        {"case": "L1 (run 2)", "pass": lt.INFRA, "summary": {"not_run_reason": "Groq rate limit reached."}},
        {"case": "L2 (run 1)", "pass": False, "summary": {"fit_score": 80}},
    ]
    lt.record_into_testing_md(results, 3, "2026-11-12T10:00:00")
    text = (tmp_path / "TESTING.md").read_text(encoding="utf-8")
    l1 = next(line for line in text.splitlines() if line.startswith("| L1 |"))
    l2 = next(line for line in text.splitlines() if line.startswith("| L2 |"))
    assert "PASS (score 86)" in l1 and "NOT RUN" in l1 and "FAIL" not in l1
    assert "FAIL (score 80)" in l2
    assert "Status: recorded 2026-11-12" in text and "NOT YET RECORDED" not in text
