"""History storage: off by default, summaries only, never raises."""
import json

import pytest

import storage

ANALYSIS = {
    "fit_score": 74,
    "score_breakdown": {"skills_coverage": 30, "experience_relevance": 20,
                        "projects_achievements": 14, "education_clarity": 10},
    "required_skills": ["Python", "SQL", "Tableau"],
    "matched_skills": [
        {"skill": "Python", "evidence": "Wrote SQL queries and cleaned order data of Anita Rao with Python.",
         "evidence_type": "applied"},
        {"skill": "SQL", "evidence": "SQL, Python, Excel", "evidence_type": "listed"},
    ],
    "missing_skills": ["Tableau"],
    "preferred_skills": [{"skill": "Docker", "status": "not shown", "evidence_type": None, "evidence": None}],
    "suggestions": ["Mention the retail dashboard project from Anita Rao's internship."],
    "score_reasoning": "Anita Rao has relevant internship experience.",
}
JD = "\n  Senior Data Analyst, Acme Corp\nWe need Python and SQL..."


@pytest.fixture
def history_file(tmp_path, monkeypatch):
    path = tmp_path / "data" / "history.jsonl"
    monkeypatch.setenv("HISTORY_FILE", str(path))
    return path


def test_st01_history_is_off_by_default_and_nothing_is_written(history_file, monkeypatch):
    monkeypatch.delenv("ENABLE_HISTORY", raising=False)
    assert storage.enabled() is False
    assert storage.save(ANALYSIS, JD, "m") == (False, "History is off.")
    assert not history_file.exists() and not history_file.parent.exists()


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True), ("on", True),
                                            ("0", False), ("", False), ("no", False), ("false", False)])
def test_st02_only_explicit_values_switch_history_on(monkeypatch, value, expected):
    monkeypatch.setenv("ENABLE_HISTORY", value)
    assert storage.enabled() is expected


def test_st03_saves_a_summary_and_never_the_resume_text_quotes_or_suggestions(history_file, monkeypatch):
    monkeypatch.setenv("ENABLE_HISTORY", "1")
    ok, message = storage.save(ANALYSIS, JD, "openai/gpt-oss-120b")
    assert ok and str(history_file) in message
    raw = history_file.read_text(encoding="utf-8")
    entry = json.loads(raw)
    assert entry["fit_score"] == 74 and entry["job_title"] == "Senior Data Analyst, Acme Corp"
    assert entry["matched"] == [{"skill": "Python", "evidence_type": "applied"},
                                {"skill": "SQL", "evidence_type": "listed"}]
    assert entry["missing"] == ["Tableau"] and entry["required_count"] == 3
    # personal data from the resume must not be in the file
    for private in ("Anita", "Rao", "cleaned order data", "retail dashboard", "internship"):
        assert private not in raw, private
    assert "evidence" not in entry and "suggestions" not in entry and "score_reasoning" not in entry
    assert '"evidence":' not in raw  # (the key "evidence_type" is fine: it holds only applied/listed/mentioned)


def test_st04_load_returns_newest_first_with_a_limit_and_skips_broken_lines(history_file, monkeypatch):
    monkeypatch.setenv("ENABLE_HISTORY", "1")
    for score in (10, 20, 30):
        storage.save({**ANALYSIS, "fit_score": score}, JD, "m")
    with history_file.open("a", encoding="utf-8") as handle:
        handle.write("this line is not json\n")
    assert [e["fit_score"] for e in storage.load()] == [30, 20, 10]
    assert [e["fit_score"] for e in storage.load(limit=2)] == [30, 20]


def test_st05_missing_file_is_an_empty_history_and_clear_works(history_file, monkeypatch):
    monkeypatch.setenv("ENABLE_HISTORY", "1")
    assert storage.load() == []
    storage.save(ANALYSIS, JD, "m")
    assert storage.clear() is True and not history_file.exists()
    assert storage.clear() is True  # clearing an already empty history is fine


def test_st06_a_write_failure_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_HISTORY", "1")
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where a folder is needed")
    monkeypatch.setenv("HISTORY_FILE", str(blocker / "history.jsonl"))
    ok, message = storage.save(ANALYSIS, JD, "m")
    assert ok is False and "Could not save" in message


def test_st07_partial_analysis_and_long_title_are_handled():
    summary = storage.summarize({"fit_score": 5}, "x" * 500, "m")
    assert summary["fit_score"] == 5 and summary["matched"] == [] and len(summary["job_title"]) == 80
    assert storage.summarize(None)["fit_score"] is None
