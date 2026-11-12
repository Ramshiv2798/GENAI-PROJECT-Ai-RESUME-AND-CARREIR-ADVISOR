"""
Optional local history of analysis SUMMARIES.
Privacy rules (enforced here and tested):
- OFF unless the environment variable ENABLE_HISTORY is 1/true/yes/on. On a public deployment
  (e.g. Streamlit Community Cloud) leave it off: the server's files are shared and temporary.
  Visitors keep their results with the Download buttons, which save to their own computer.
- Only a summary is stored: date, job title line, scores, skill NAMES and how each was matched.
  The resume text, the evidence quotes (they come from the resume) and the suggestions are never stored.
- Stored as one JSON object per line in data/history.jsonl (change with HISTORY_FILE).
"""
import datetime
import json
import os
import uuid
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "history.jsonl"
MAX_TITLE_CHARS = 80


def enabled():
    """True only when history was deliberately switched on."""
    return os.getenv("ENABLE_HISTORY", "").strip().lower() in ("1", "true", "yes", "on")


def history_path():
    configured = os.getenv("HISTORY_FILE", "").strip()
    return Path(configured) if configured else DEFAULT_PATH


def job_title(job_description):
    """First non-empty line of the job description, shortened (it is usually the job title)."""
    for line in (job_description or "").splitlines():
        line = line.strip()
        if line:
            return line[:MAX_TITLE_CHARS]
    return ""


def summarize(analysis, job_description="", model=""):
    """The part of an analysis that is safe to keep. Works with partial dictionaries."""
    analysis = analysis or {}
    return {
        "id": uuid.uuid4().hex[:8],
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "job_title": job_title(job_description),
        "model": model,
        "fit_score": analysis.get("fit_score"),
        "score_breakdown": analysis.get("score_breakdown", {}),
        "required_count": len(analysis.get("required_skills", [])),
        "matched": [
            {"skill": m.get("skill"), "evidence_type": m.get("evidence_type")}
            for m in analysis.get("matched_skills", [])
        ],
        "missing": list(analysis.get("missing_skills", [])),
        "preferred": [
            {"skill": p.get("skill"), "status": p.get("status")}
            for p in analysis.get("preferred_skills", [])
        ],
    }


def save(analysis, job_description="", model=""):
    """Append one summary. Returns (True, message) or (False, reason); never raises."""
    if not enabled():
        return False, "History is off."
    try:
        path = history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summarize(analysis, job_description, model), ensure_ascii=False) + "\n")
        return True, f"Summary saved to {path}"
    except OSError as error:
        return False, f"Could not save the history ({type(error).__name__}). The analysis itself is not affected."


def load(limit=50):
    """Newest first. Unreadable lines are skipped; a missing file means an empty history."""
    path = history_path()
    if not path.is_file():
        return []
    entries = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    except OSError:
        return []
    return list(reversed(entries))[:limit]


def clear():
    """Delete the history file. Returns True if the history is now empty."""
    try:
        history_path().unlink(missing_ok=True)
        return True
    except OSError:
        return False
