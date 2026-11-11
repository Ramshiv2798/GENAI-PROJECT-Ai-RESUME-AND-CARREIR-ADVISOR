"""Interface tests (Streamlit AppTest). The model and Groq are mocked.

Run:  pytest -v
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core

ROOT = Path(__file__).resolve().parent.parent
APP_FILE = str(ROOT / "app.py")
DEFAULT_REPORT = "Upload a resume and job description, then click Analyze."

RESUME = "Anita Rao. B.Tech CSE 2024. Skills: Python, SQL, Tableau, Excel. Data analyst intern for 6 months."
JD = "Data Analyst. Required: Python, SQL, Tableau, statistics, A/B testing."
OTHER_RESUME = "Vikram Singh. Warehouse supervisor for eight years, forklift certified, inventory control."

GOOD_STATE = {"resume": RESUME, "jd": JD, "analysis": {"fit_score": 74}, "warnings": []}
GOOD_REPORT = "## Fit Score: 74/100 (Moderate fit)"
GOOD_RESULT = {"fit_score": 74}

HAS_UPLOAD_SUPPORT = hasattr(AppTest, "file_uploader")


def new_app():
    at = AppTest.from_file(APP_FILE, default_timeout=15)
    at.run()
    assert not at.exception
    return at


def button(at, label):
    return next(b for b in at.button if b.label == label)


def analyze(at, resume=RESUME, jd=JD):
    at.text_area(key="resume_text").set_value(resume)
    at.text_area(key="job_desc").set_value(jd)
    button(at, "Analyze").click().run()
    assert not at.exception
    return at


def ask(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception
    return at


def all_markdown(at):
    return "\n".join(m.value for m in at.markdown)


@pytest.fixture
def good_analysis(monkeypatch):
    monkeypatch.setattr(core, "safe_analyze", lambda f, t, j: (GOOD_STATE, GOOD_REPORT, GOOD_RESULT))


# ---------------- UI01: first load ----------------
def test_ui01_initial_page():
    at = new_app()
    assert at.title[0].value == "AI Resume Screener and Career Advisor"
    assert DEFAULT_REPORT in all_markdown(at)
    assert at.session_state["state"] is None
    assert at.session_state["messages"] == []
    assert len(at.chat_message) == 0


# ---------------- UI02: successful analysis ----------------
def test_ui02_analyze_shows_report_and_json(good_analysis):
    at = analyze(new_app())
    assert at.session_state["state"] == GOOD_STATE
    assert "Fit Score: 74/100" in all_markdown(at)
    assert len(at.json) == 1


def test_ui02b_analyze_passes_inputs_to_core(monkeypatch):
    seen = {}

    def fake(file_path, pasted_text, job_description):
        seen.update(file_path=file_path, resume=pasted_text, jd=job_description)
        return GOOD_STATE, GOOD_REPORT, GOOD_RESULT

    monkeypatch.setattr(core, "safe_analyze", fake)
    analyze(new_app())
    assert seen == {"file_path": None, "resume": RESUME, "jd": JD}  # no upload -> no path


# ---------------- UI03: failed analysis shows the error and keeps no state ----------------
def test_ui03_failed_analysis_clears_state_and_chat(monkeypatch):
    monkeypatch.setattr(core, "safe_analyze", lambda f, t, j: (GOOD_STATE, GOOD_REPORT, GOOD_RESULT))
    monkeypatch.setattr(core, "chat_stream", lambda m, h, s: iter(["Focus on your A/B test project."]))
    at = analyze(new_app())
    ask(at, "What should I improve?")
    assert len(at.session_state["messages"]) == 2

    # second analysis fails: core returns state=None, like safe_analyze does on any error
    monkeypatch.setattr(core, "safe_analyze", lambda f, t, j: (None, "**Error:** boom", None))
    analyze(at, resume="Another candidate " * 10)
    assert at.session_state["state"] is None
    assert at.session_state["messages"] == []  # the old chat does not survive
    assert len(at.chat_message) == 0
    assert "**Error:** boom" in all_markdown(at)
    assert len(at.json) == 0


# ---------------- UI03b: ANY failure path must leave the advisor without old data ----------------
def _returns_none(f, t, j):
    return None, "**Error:** model failed", None


def _raises_oserror(f, t, j):
    raise OSError("disk full while saving the upload")


def _raises_unexpected(f, t, j):
    raise KeyError("something nobody planned for")


@pytest.mark.parametrize("failing", [_returns_none, _raises_oserror, _raises_unexpected])
def test_ui03b_advisor_never_sees_an_earlier_candidate_after_any_failure(monkeypatch, failing):
    seen_states = []

    def fake_stream(message, history, state):
        seen_states.append(state)
        yield "ok"

    monkeypatch.setattr(core, "chat_stream", fake_stream)
    monkeypatch.setattr(core, "safe_analyze", lambda f, t, j: (GOOD_STATE, GOOD_REPORT, GOOD_RESULT))
    at = analyze(new_app())
    ask(at, "first question about Anita")
    assert seen_states == [GOOD_STATE]

    # re-analysis for a different candidate fails in the given way
    monkeypatch.setattr(core, "safe_analyze", failing)
    analyze(at, resume=OTHER_RESUME)
    assert at.session_state["state"] is None
    assert at.session_state["result"] is None
    assert at.session_state["messages"] == []
    assert "Error" in all_markdown(at)  # the user is told, not left with a stale report
    assert "Fit Score: 74/100" not in all_markdown(at)

    ask(at, "What should I improve?")
    assert seen_states == [GOOD_STATE]  # chat_stream was NOT called again: no old data reachable
    assert at.session_state["messages"][-1] == {
        "role": "assistant",
        "content": "Please analyze a resume and job description first.",
    }


def test_ui03c_old_analysis_is_cleared_before_the_new_one_starts(monkeypatch):
    """The wipe happens first: while the new analysis runs, the old state is already gone."""
    monkeypatch.setattr(core, "safe_analyze", lambda f, t, j: (GOOD_STATE, GOOD_REPORT, GOOD_RESULT))
    at = analyze(new_app())
    ask_state = at.session_state["state"]
    assert ask_state == GOOD_STATE

    observed = {}

    def spy(f, t, j):
        # runs inside the callback; read what the page state looks like at this moment
        import streamlit as st
        observed["state"] = st.session_state.state
        observed["messages"] = list(st.session_state.messages)
        return None, "**Error:** boom", None

    monkeypatch.setattr(core, "safe_analyze", spy)
    analyze(at, resume=OTHER_RESUME)
    assert observed == {"state": None, "messages": []}


@pytest.mark.skipif(not HAS_UPLOAD_SUPPORT, reason="this Streamlit version's AppTest cannot upload files")
def test_ui03d_upload_save_failure_is_reported_and_clears_state(monkeypatch):
    monkeypatch.setattr(core, "safe_analyze", lambda f, t, j: (GOOD_STATE, GOOD_REPORT, GOOD_RESULT))
    at = analyze(new_app())
    assert at.session_state["state"] is not None

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(core.tempfile, "NamedTemporaryFile", broken)
    at.file_uploader[0].upload("resume.txt", RESUME.encode(), "text/plain")
    button(at, "Analyze").click().run()
    assert not at.exception
    assert at.session_state["state"] is None
    assert at.session_state["messages"] == []
    assert "could not be saved" in all_markdown(at)


@pytest.mark.skipif(not HAS_UPLOAD_SUPPORT, reason="this Streamlit version's AppTest cannot upload files")
def test_ui03e_uploaded_file_reaches_core_and_temp_file_is_deleted(monkeypatch):
    seen = {}

    def fake(file_path, pasted_text, job_description):
        seen["path"] = file_path
        seen["text"] = Path(file_path).read_text()
        return GOOD_STATE, GOOD_REPORT, GOOD_RESULT

    monkeypatch.setattr(core, "safe_analyze", fake)
    at = new_app()
    at.file_uploader[0].upload("resume.txt", RESUME.encode(), "text/plain")
    at.text_area(key="job_desc").set_value(JD)
    button(at, "Analyze").click().run()
    assert not at.exception
    assert seen["text"] == RESUME and seen["path"].endswith(".txt")
    assert not Path(seen["path"]).exists()


# ---------------- UI04: chat ----------------
def test_ui04_chat_before_analysis_is_refused(monkeypatch):
    called = []
    monkeypatch.setattr(core, "chat_stream", lambda m, h, s: called.append(1) or iter(["x"]))
    at = ask(new_app(), "hello")
    assert called == []  # the model is never called without an analysis
    msgs = at.session_state["messages"]
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1] == {"role": "assistant", "content": "Please analyze a resume and job description first."}


def test_ui04b_chat_streams_and_keeps_final_text(good_analysis, monkeypatch):
    monkeypatch.setattr(core, "chat_stream", lambda m, h, s: iter(["Focus", "Focus on your", "Focus on your A/B test."]))
    at = ask(analyze(new_app()), "What should I improve?")
    assert at.session_state["messages"][-1] == {"role": "assistant", "content": "Focus on your A/B test."}
    assert [m.name for m in at.chat_message] == ["user", "assistant"]


def test_ui04c_history_excludes_the_new_question(good_analysis, monkeypatch):
    seen = []

    def fake_stream(message, history, state):
        seen.append((message, [dict(m) for m in history], state))
        yield f"answer to {message}"

    monkeypatch.setattr(core, "chat_stream", fake_stream)
    at = analyze(new_app())
    ask(at, "first question")
    ask(at, "second question")

    assert seen[0][0] == "first question" and seen[0][1] == []
    assert seen[1][0] == "second question"
    assert seen[1][1] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer to first question"},
    ]
    assert seen[1][2] == GOOD_STATE  # the advisor gets the current analysis


def test_ui04d_chat_error_is_shown_politely(good_analysis, monkeypatch):
    def broken(message, history, state):
        yield "partial"
        raise RuntimeError("Groq rate limit reached.")

    monkeypatch.setattr(core, "chat_stream", broken)
    at = ask(analyze(new_app()), "hi")
    reply = at.session_state["messages"][-1]["content"]
    assert reply.startswith("Sorry, I could not answer that.") and "rate limit" in reply


def test_ui04e_blank_question_is_ignored(good_analysis):
    at = ask(analyze(new_app()), "   ")
    assert at.session_state["messages"] == []


# ---------------- UI05: clear all ----------------
def test_ui05_clear_all_resets_everything(good_analysis, monkeypatch):
    monkeypatch.setattr(core, "chat_stream", lambda m, h, s: iter(["ok"]))
    at = ask(analyze(new_app()), "hello")
    assert at.session_state["state"] is not None

    button(at, "Clear all").click().run()
    assert not at.exception
    assert at.session_state["state"] is None
    assert at.session_state["messages"] == []
    assert at.session_state["result"] is None
    assert at.text_area(key="resume_text").value == ""
    assert at.text_area(key="job_desc").value == ""
    assert DEFAULT_REPORT in all_markdown(at)
    assert len(at.json) == 0
    assert len(at.chat_message) == 0


# =====================================================================================
# UI06: editing the inputs after an analysis must invalidate it (review round 2)
# =====================================================================================
STATE_A = {"resume": RESUME, "jd": JD, "analysis": {"fit_score": 74}, "warnings": []}
STATE_B = {"resume": OTHER_RESUME, "jd": JD, "analysis": {"fit_score": 12}, "warnings": []}


@pytest.fixture
def two_candidates(monkeypatch):
    """safe_analyze returns candidate A's or B's analysis depending on the pasted text; the
    advisor records which candidate's state it was given."""
    seen_by_advisor = []

    def fake_analyze(file_path, pasted_text, job_description):
        if "Anita" in (pasted_text or "") or "Anita" in (Path(file_path).read_text() if file_path else ""):
            return STATE_A, "## Fit Score: 74/100 (Moderate fit) - Anita", {"fit_score": 74}
        return STATE_B, "## Fit Score: 12/100 (Weak fit) - Vikram", {"fit_score": 12}

    def fake_stream(message, history, state):
        seen_by_advisor.append(state)
        yield "Answer about " + state["resume"][:5]

    monkeypatch.setattr(core, "safe_analyze", fake_analyze)
    monkeypatch.setattr(core, "chat_stream", fake_stream)
    return seen_by_advisor


def test_ui06_editing_the_resume_clears_the_old_analysis_and_chat(two_candidates):
    at = analyze(new_app())                      # Candidate A analysed
    ask(at, "What should I improve?")
    assert at.session_state["state"] == STATE_A and len(at.session_state["messages"]) == 2

    # the reviewer's sequence: replace the pasted text with Candidate B, do NOT click Analyze
    at.text_area(key="resume_text").set_value(OTHER_RESUME).run()
    assert not at.exception
    assert at.session_state["state"] is None
    assert at.session_state["result"] is None
    assert at.session_state["messages"] == []
    assert len(at.chat_message) == 0
    assert "Anita" not in all_markdown(at)       # Candidate A's report is gone from the page
    assert DEFAULT_REPORT in all_markdown(at)
    assert any("changed" in i.value for i in at.info)

    # the advisor cannot answer from Candidate A any more
    ask(at, "And what about my skills?")
    assert two_candidates == [STATE_A]           # advisor was called only before the edit
    assert at.session_state["messages"][-1]["content"] == "Please analyze a resume and job description first."


def test_ui06b_editing_the_job_description_also_invalidates(two_candidates):
    at = analyze(new_app())
    at.text_area(key="job_desc").set_value(JD + " Required: Docker.").run()
    assert not at.exception
    assert at.session_state["state"] is None and at.session_state["messages"] == []
    assert DEFAULT_REPORT in all_markdown(at)


def test_ui06c_unchanged_inputs_keep_the_analysis_and_chat(two_candidates):
    at = analyze(new_app())
    ask(at, "first question")
    ask(at, "second question")                   # every interaction reruns the script
    assert at.session_state["state"] == STATE_A
    assert len(at.session_state["messages"]) == 4
    assert two_candidates == [STATE_A, STATE_A]
    assert not at.info


def test_ui06d_after_invalidation_a_new_analysis_works_and_chat_uses_it(two_candidates):
    at = analyze(new_app())
    at.text_area(key="resume_text").set_value(OTHER_RESUME).run()
    assert at.session_state["state"] is None
    button(at, "Analyze").click().run()
    assert not at.exception
    assert at.session_state["state"] == STATE_B
    assert not at.info                           # the "inputs changed" notice is gone
    ask(at, "How do I improve?")
    assert two_candidates == [STATE_B]


def test_ui06e_reverting_the_edit_does_not_resurrect_the_old_analysis(two_candidates):
    at = analyze(new_app())
    at.text_area(key="resume_text").set_value(OTHER_RESUME).run()
    at.text_area(key="resume_text").set_value(RESUME).run()   # typed the original text back
    assert at.session_state["state"] is None                  # must analyze again, nothing stale returns
    assert at.session_state["messages"] == []


@pytest.mark.skipif(not HAS_UPLOAD_SUPPORT, reason="this Streamlit version's AppTest cannot upload files")
def test_ui06f_replacing_the_uploaded_file_invalidates(two_candidates):
    at = new_app()
    at.file_uploader[0].upload("a.txt", RESUME.encode(), "text/plain")
    at.text_area(key="job_desc").set_value(JD)
    button(at, "Analyze").click().run()
    assert at.session_state["state"] == STATE_A

    at.file_uploader[0].upload("b.txt", OTHER_RESUME.encode(), "text/plain").run()
    assert not at.exception
    assert at.session_state["state"] is None and at.session_state["messages"] == []
    assert DEFAULT_REPORT in all_markdown(at)


def test_ui07_unexpected_error_details_are_not_shown_on_the_page(monkeypatch):
    def secret_failure(f, t, j):
        raise KeyError("secret internal detail")
    monkeypatch.setattr(core, "safe_analyze", secret_failure)
    at = analyze(new_app())
    assert at.session_state["state"] is None
    page = all_markdown(at)
    assert "secret internal detail" not in page and "could not be completed" in page
