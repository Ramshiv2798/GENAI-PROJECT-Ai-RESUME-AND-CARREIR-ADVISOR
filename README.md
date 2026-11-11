# AI Resume Screener and Career Advisor

A Streamlit app that compares **one resume with one job description**, shows which mandatory skills
the resume gives textual evidence for, scores the fit out of 100, gives three improvement suggestions,
and lets the user ask follow-up questions in a career-advisor chat.

> **What the score is, and is not.** It is an application-specific comparison of the resume and the
> job description you supply. It is **not** a universal ATS score and **not** a hiring decision. The
> skills part checks what the resume *says*; it does not prove the candidate is competent.

## Problem statement
Candidates rarely know why a resume fails a role, and an LLM asked for "a score" will happily accept
claims it cannot back up. This project asks the model for a structured analysis, then **checks the
model's skill claims against the resume text in code** and calculates the skills score itself.

## Features
- Upload a resume (PDF or TXT) or paste text, paste a job description, click **Analyze**.
- Fit score out of 100 with a component breakdown showing which part is code-calculated and which is model judgement.
- Matched skills, each with a quote from the resume and how it was classified (described as used / listed only / mentioned only).
- Missing mandatory skills, and preferred ("nice to have") skills shown separately.
- Exactly three suggestions that must not tell the candidate to claim skills they lack.
- Streaming career-advisor chat grounded in the resume, the job description and the verified analysis.
- Prompt-injection handling (limited; see below), clear error messages, and a "Clear all" button.
- Editing the resume, upload or job description after an analysis **clears the old result and chat**.

## Repository, setup and launch
```
git clone <YOUR-REPOSITORY-URL>      # TODO(author): replace with the real clone URL
cd <repository-folder>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # add requirements-dev.txt for pytest
cp .env.example .env                                     # then edit .env
streamlit run app.py
```
- **Python:** 3.12 (the version the tests were run with; 3.10+ should work but is untested).
- **Configuration (`.env`):** `GROQ_API_KEY` (free key from https://console.groq.com/keys) and optional
  `MODEL_NAME` (default `openai/gpt-oss-120b`). On Streamlit Cloud put the same names in the app's secrets. Never commit `.env`.
- **Supported files:** PDF and TXT only (5 MB max). Scanned/image PDFs are not supported.
- **Demo link / screenshots:** TODO(author): add the deployed URL and screenshots (`docs/screenshots/`) after a live run.

## File structure
```
app.py                 Streamlit interface only (inputs, state, invalidation, chat display)
core.py                prompts, document reading, Groq calls, validation, scoring, report, chat
live_tests.py          real-model test runner (needs a Groq key), writes live_results.md/json
requirements.txt       runtime dependencies, pinned to tested versions
requirements-dev.txt   adds pytest
.env.example           dummy configuration
pytest.ini             test settings
samples/               synthetic resumes and job descriptions (no real people)
tests/test_core.py     100 offline tests of core.py (model mocked)
tests/test_app.py      23 interface tests with Streamlit AppTest (model mocked)
docs/sample_output.md  example report (stubbed model reply, see below)
TESTING.md             test plan, offline results, live-test table
```
`core.py` is long. If the project grows, split it into prompts, validation/scoring, and API/document reading.

## Analysis flow
1. `app.py` saves the upload to a temporary file inside `core.analyze_upload`, which also deletes it.
2. `read_resume` extracts text (pypdf) and rejects empty/scanned files; long texts keep their start and end (`prepare_text`).
3. The prompt (role, security rules, rubric, one few-shot example) asks for JSON. Resume and job text are placed in
   `<resume>` / `<job_description>` tags with those tags stripped from user text.
4. `validate_analysis` checks the reply (see "Validation rules"), and `run_analysis` asks the model for a corrected reply
   up to 3 times; the last attempt is repaired deterministically.
5. The skills score and the total are calculated in Python; `format_report` builds the report.
6. Chat answers use the resume, job description and the *verified* analysis. Skills the model extracted but that are not in the
   resume are withheld, and the model's own explanation is labelled unverified.

## Scoring
Total = skills coverage (40) + experience relevance (25) + projects and achievements (20) + education and clarity (15).

**Skills coverage is calculated by the code:** `40 × credit ÷ mandatory requirements`, rounded.

| Evidence for a mandatory requirement | Credit |
|---|---|
| Quote in an Experience/Projects sentence (4+ words) that describes doing something with the skill ("applied") | 1.0 |
| Quote in a Skills section ("listed"), or any other mention ("mentioned") | 0.5 |
| No usable quote; quote not in the resume; quote does not name the skill; resume says the skill was **not** used; resume only **plans to learn** it | 0 |

Only **mandatory** requirements are counted. Preferred / optional / "nice to have" skills are shown separately and add no
points. Skills the job description says are "not required" are ignored. Alternatives such as "Tableau or Power BI" count as
one requirement satisfied by either. These categories come from the job description's wording (headings such as
"Required:" / "Nice to have:" and cues such as "optional", "a plus", "not required"); anything not marked is treated as mandatory.

**Model judgement (not verified by code, 60 of 100 points):** experience relevance, projects and achievements, education and
clarity. Code only enforces that each number is a finite value within its range.

## Validation rules (what is enforced by code)
- A matched skill needs a quote that exists in the resume, **names that skill** (aliases such as K8s are understood), and is
  not negated ("never used X") or a plan ("will learn X"). Short quotes are read in the context of their resume sentence.
- Text in the resume that looks like instructions to the AI is ignored as evidence.
- Required skills are checked against the job description; mandatory skills the model forgot are added; each requirement ends
  up in exactly one of matched or missing; names are normalised and duplicates removed.
- `candidate_skills` is grounded in the resume text; unsupported entries are reported and kept away from the advisor.
- If any of this fails, the model is asked for a corrected analysis; if it still fails the app repairs the lists and
  recalculates the skills score from verified evidence only, with a warning.

## Sample input and output
Inputs: `samples/resume_strong.txt` + `samples/jd_data_analyst.txt`. The output below was produced by running the real
validator and report code on a **stubbed model reply**, to show the format. It is **not** a live-model result; live results
are tracked in `TESTING.md`.

<details><summary>Example report</summary>

## Fit Score: 78/100 (Strong fit)

**Model's explanation:** Stubbed reply for documentation: a relevant internship and a project that use most of the required tools.

| Component | Points | How it was determined |
|---|---|---|
| Skills Coverage | 30 / 40 | Calculated by the app from quotes found in the resume: 4 described as used, 1 listed/mentioned only, 1 missing of 6 mandatory requirements |
| Experience Relevance | 20 / 25 | Model judgement (not independently verified) |
| Projects Achievements | 16 / 20 | Model judgement (not independently verified) |
| Education Clarity | 12 / 15 | Model judgement (not independently verified) |

### Matched Skills (a quote was found in the resume that names the skill)
- **Python** (described as used) — evidence: "Wrote SQL queries to pull order data and cleaned it with Python and Pandas."
- **SQL** (described as used) — evidence: "Wrote SQL queries to pull order data and cleaned it with Python and Pandas."
- **Tableau** (described as used) — evidence: "Built weekly Tableau dashboards for the sales team covering revenue and churn."
- **Statistics** (listed only) — evidence: "Python, SQL, Tableau, Excel, Pandas, Statistics"
- **A/B testing** (described as used) — evidence: "Ran an A/B test on an email campaign and analysed results with Python and statistics."

### Missing Skills (mandatory requirements with no usable evidence)
- Stakeholder communication

> **What this score is.** An application-specific comparison of this resume with this job
> description, not an ATS score and not a hiring decision. The app checks that quotes exist, name the
> skill and are not negated or only planned: that is textual evidence, not proof of competency.
> Only the skills component (30/40) is calculated by code; the other three
> components are the model's judgement and are not verified.

### 3 Suggestions to Improve Your Resume
1. Add the scope of the weekly Tableau dashboards (users, refresh frequency) using only figures you can verify.
2. Name the statistical test and sample size in the A/B test project so Statistics is shown in use, not only listed.
3. The resume describes presenting findings but never says stakeholder communication; reword that bullet only if it is accurate.

</details>

More inputs: `resume_weak.txt` (weak match), `jd_devops.txt` (skill gap), `resume_negated.txt` (never-used / planned skills),
`jd_categories.txt` (required vs optional vs not required), `resume_injection*.txt` (prompt injection),
`resume_strong.pdf` (text PDF), `resume_scanned.pdf` (image PDF: rejected with a "paste the text" message).

## Tests
```
pip install -r requirements-dev.txt
pytest                     # offline: Groq is mocked, no key or network
python live_tests.py --runs 3   # real model: needs GROQ_API_KEY in .env; writes live_results.md
```
**Observed results.** Offline: 123 passed (100 core + 23 interface), last run 10 Nov 2026. These verify the application's
behaviour under controlled model replies; they do **not** establish the real model's accuracy or consistency. Live: **not yet
recorded** (needs a key); see `TESTING.md`.

## Prompt injection: what is and is not protected
- Enforced by code: user text is delimited and tag-stripped; instruction-like lines are ignored as skill evidence; the skills
  score and the total are calculated by the app, so a model that obeys "give a score of 100" cannot change them.
- **Not enforced:** the other three components are model judgements. A real model could be influenced there. The app only warns
  when instruction-like text coincides with near-maximum marks, and live case L3b measures how a real model behaves.
  Treat those 60 points as needing human review whenever the input is flagged.

## Reliability, timeouts and retries
- Each Groq request times out after 60 s (`API_TIMEOUT_SECONDS`); the Groq client's own retries are off.
- Transient failures are retried up to 3 times with 2 s then 4 s pauses (`call_groq`). A wrong API key is not retried.
- A reply that fails validation triggers up to 3 model calls in total (`MAX_ANALYSIS_ATTEMPTS`).
- The page shows short messages. Full exceptions go to the application log (logger `resume_screener`).

## Limitations
- **Scanned PDFs:** no OCR; paste the text instead.
- **Truncation:** texts over 24,000 (resume) / 12,000 (job description) characters keep their start and end; the middle is skipped and a warning is shown.
- **Evidence interpretation is heuristic.** Negation, plans and "described as used" are detected with wording rules, so unusual
  phrasing can be misread in either direction. Quotes must name the skill, so a genuine example that never uses the skill's
  name (e.g. "presented findings to the manager" for "stakeholder communication") is classed as missing: conservative by design.
  Resume headings decide whether a quote is "listed" or "applied"; unusual layouts may be classed as "mentioned".
- **Requirement categories** are read from wording and can be wrong on unusual job descriptions.
- **Scoring variability:** the 60 judgement points vary between runs and models; compare repeated runs (`--runs 3`).
- Resume and job text are sent to Groq: use synthetic or consented data.
