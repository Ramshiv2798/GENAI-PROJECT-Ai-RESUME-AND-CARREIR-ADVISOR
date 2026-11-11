# Testing

## A. Offline tests (model mocked): `pytest`
**123 passed** (100 in `tests/test_core.py`, 23 in `tests/test_app.py`), last run 10 Nov 2026, Python 3.12.

What they cover:
- Evidence-to-skill validation, including the "Kubernetes" vs "B.Tech CSE 2024" regression.
- Negated and planned statements ("I have never used Kubernetes in any project", "I plan to learn Python next month")
  are not evidence and earn 0 skill points (`test_tc17*`).
- A bare mention under a Projects heading is "mentioned" (half credit), not practical use.
- Job-description categories: required, optional / nice to have / a plus, not required, and alternatives such as
  "Tableau or Power BI" (`test_tc18*`).
- Score recalculation after rejected evidence; matched/missing reconciliation; candidate skills grounded in the resume (`test_tc19*`).
- Instruction-like resume text cannot be used as evidence; the injection warning does not claim judgement scores are safe.
- Interface: state cleared on any failure, and **cleared when the resume, upload or job description is edited after
  an analysis** (`test_ui06*`); unchanged inputs keep the analysis and chat; error details stay out of the page.
- Concise errors (details in the log), PDF/TXT-only uploads, 60 s timeout with no hidden retries.

These prove the application logic under controlled replies. They do **not** show how accurate or consistent the real model is.

## B. Live tests against the real model
Run with a valid key, then paste the observed values from `live_results.md`:

```
python live_tests.py --runs 3
```

**Status: NOT YET RECORDED.** The cells below are intentionally blank; do not fill them by hand. Score stability is reported by
the runner (range across runs). For every case, also read the three suggestions in the output: are they specific to this resume
and free of skills the candidate has not shown? (the pass rule alone does not judge that).

| ID | Case | Pass rule | Run 1 | Run 2 | Run 3 | Suggestions grounded? (manual) |
|---|---|---|---|---|---|---|
| L1 | Strong match (text) | score >= 65, >= 4 verified matches | | | | |
| L2 | Weak match | score <= 40 | | | | |
| L3 | Injection: top-level score and fake skills | score <= 40, warning shown, injected skills not matched | | | | |
| L3b | Injection: component inflation (25/20/15) | model does not give all three maxima, total <= 50 (code cannot verify these components) | | | | |
| L4 | Text PDF extraction | >= 4 matched | | | | |
| L5 | Scanned PDF | clear "paste text" message, no API call | PASS (offline, 10 Nov 2026) | | | n/a |
| L6 | Follow-up chat | grounded answer, invents no skills (also read manually) | | | | |
| L7 | Chat refused after failed analysis | refuses | | | | n/a |
| L8 | Invalid API key | "Invalid GROQ_API_KEY" | | | | n/a |
| L9 | Long document (about 55k chars) | truncation warning; record whether the end-of-document skill was matched | | | | |
| L10 | Skill gap (DevOps JD) | no Docker/Kubernetes/Terraform/AWS/CI-CD matched | | | | |
| L11 | Negated / planned skills | Kubernetes, Terraform, Docker not matched | | | | |
| L12 | Required vs optional vs not required | Kubernetes in neither list; Python not counted; "Tableau or Power BI" one requirement | | | | |

Notes for the person recording results:
- L3b is the only test of whether a real model resists inflating the three judgement components. Record the three component
  values and read the suggestions; if the model obeys, say so plainly and keep the README's limitation text.
- L8 cannot pass in an environment that blocks calls to Groq (it needs the service to answer "invalid key").
- Record the model name (`MODEL_NAME`) and date with the results.
