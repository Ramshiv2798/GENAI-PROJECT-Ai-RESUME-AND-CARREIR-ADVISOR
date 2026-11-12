# AI Resume Screener and Career Advisor

A Streamlit web app that compares **one resume with one job description**. The user uploads or pastes a
resume and a job description and receives a fit score out of 100, a list of matched and missing skills
(each match backed by a quote from the resume), three improvement suggestions, and a career-advisor chat.

> **What the score is, and is not.** It is an application-specific comparison of the resume and the job
> description you supply. It is **not** a universal ATS score and **not** a hiring decision. The skills part
> checks what the resume *says*; it does not prove the candidate is competent.

## Problem statement
Candidates rarely know why a resume fails a role, and an LLM asked for "a score" will happily accept claims
it cannot back up. This project asks the model for a structured analysis, then **checks the model's skill
claims against the resume text in code** and calculates the skills score itself.

## Features
| Module | What it does |
|---|---|
| Resume and job input | Upload a resume (PDF or TXT, 5 MB max) or paste text, paste a job description, click **Analyze** |
| Fit score | Score out of 100 with a component breakdown showing which part is code-calculated and which is model judgement |
| Skill matching | Matched skills, each with a quote from the resume and a classification (described as used / listed only / mentioned only) |
| Gap analysis | Missing mandatory skills, plus preferred ("nice to have") skills shown separately |
| Improvement suggestions | Exactly three suggestions that must not tell the candidate to claim skills they lack |
| Career-advisor chat | Streaming follow-up chat grounded in the resume, the job description and the verified analysis |
| Safety and usability | Prompt-injection handling, clear error messages, a **Clear all** button. Editing the resume, upload or job description clears the old result and chat |

## Course concepts used
- **Prompt engineering** with a role, security rules, a scoring rubric and strict JSON format
- **Few-shot prompting**: one worked example is included in the analysis prompt
- **Structured JSON output** with strict validation and automatic retry (up to 3 attempts)
- **Grounded verification**: every matched skill needs a quote that exists in the resume, names the skill, and is not negated or only planned
- **Code-calculated scoring**: the skills score and the total are computed in Python, not trusted from the model
- **Streaming** responses for the career-advisor chat
- **Prompt-injection defence**: user text is delimited in tags, tags are stripped, and instruction-like lines are ignored as evidence

## Project structure
```
app.py                 Streamlit interface only (inputs, state, invalidation, chat display)
core.py                Prompts, document reading, Groq calls, validation, scoring, report, chat
live_tests.py          Real-model test runner (needs a Groq key), writes live_results.md/json
requirements.txt       Runtime dependencies, pinned to tested versions
requirements-dev.txt   Adds pytest
.env.example           Template for your API key
pytest.ini             Test settings
samples/               Synthetic resumes and job descriptions (no real people)
tests/test_core.py     116 offline tests of core.py (model mocked)
tests/test_app.py      23 interface tests with Streamlit AppTest (model mocked)
docs/sample_output.md  Example report (stubbed model reply, see below)
TESTING.md             Test plan, offline results, live-test table
```
`core.py` is long. If the project grows, split it into prompts, validation/scoring, and API/document reading.

## Setup
1. Install **Python 3.11**. All tests were run with Python 3.11+; other versions have not been tested.
2. Create a free Groq API key at https://console.groq.com/keys
3. Get the project and open a terminal in the project folder (the one that contains `app.py`):
   - If you received a ZIP, unzip it.
   - If it is on GitHub, press the green **Code** button, copy the HTTPS link, run `git clone` followed by that link,
     then `cd` into the new folder.
4. Create a virtual environment and install the dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate               # Windows   (Mac/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
5. Copy `.env.example` to `.env` (Windows: `copy .env.example .env`, Mac/Linux: `cp .env.example .env`) and put your key in it:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
6. Run the app:
   ```
   streamlit run app.py
   ```
   It opens at http://localhost:8501

Supported files: PDF and TXT only (5 MB max). Scanned or image PDFs are not supported; paste the text instead.

### Configuration
| Setting | Required | Default |
|---|---|---|
| `GROQ_API_KEY` | yes | none |
| `MODEL_NAME` | no | `openai/gpt-oss-120b` |

Settings are read from `.env` locally or from Streamlit secrets when deployed. Never commit `.env`.

## Run the tests
```
pip install -r requirements-dev.txt
pytest                                    # offline: Groq is mocked, no key or network needed
python live_tests.py --runs 3 --record    # real model: needs GROQ_API_KEY and internet access to Groq
```
The live command writes `live_results.md` (full outputs and suggestions) and fills the live-test table in `TESTING.md`
from the real results. It first makes one small test call, and if the key or network is wrong it stops without recording anything.

**Observed results.** Offline: **139 passed** (116 core + 23 interface), last run 12 Nov 2026 on Python 3.12.3. These verify the
application's behaviour under controlled model replies; they do **not** establish the real model's accuracy or consistency.
Live: **not yet recorded.** It needs a Groq key and network access; see `TESTING.md`.

## How the analysis works
1. `app.py` saves the upload to a temporary file inside `core.analyze_upload`, which also deletes it.
2. `read_resume` extracts text (pypdf) and rejects empty or scanned files. Long texts keep their start and end (`prepare_text`).
3. The prompt (role, security rules, rubric, one few-shot example) asks for JSON. Resume and job text are placed in
   `<resume>` / `<job_description>` tags, with those tags stripped from user text.
4. `validate_analysis` checks the reply, and `run_analysis` asks the model for a corrected reply up to 3 times;
   the last attempt is repaired deterministically.
5. The skills score and the total are calculated in Python, and `format_report` builds the report.
6. Chat answers use the resume, the job description and the *verified* analysis. Skills the model extracted but that are
   not in the resume are withheld, and the model's own explanation is labelled unverified.

## How the score is calculated
Total = skills coverage (40) + experience relevance (25) + projects and achievements (20) + education and clarity (15).

**Skills coverage is calculated by the code:** `40 x credit / mandatory requirements`, rounded.

| Evidence for a mandatory requirement | Credit |
|---|---|
| Quote in an Experience/Projects sentence (4+ words) that describes doing something with the skill ("applied") | 1.0 |
| Quote in a Skills section ("listed"), or any other mention ("mentioned") | 0.5 |
| No usable quote; quote not in the resume; quote does not name the skill; resume says the skill was **not** used; resume only **plans to learn** it | 0 |

Rules in detail:
- Only **mandatory** requirements are counted. Preferred / optional / "nice to have" skills are shown separately and add no points.
- Skills the job description says are "not required" are ignored.
- Alternatives such as "Tableau or Power BI" count as one requirement, satisfied by either.
- Categories come from the job description's wording (headings such as "Required:" / "Nice to have:" and cues such as
  "optional", "a plus", "not required"). Anything not marked is treated as mandatory.
- The wording is read **clause by clause**, so one sentence can mix categories: "SQL is required, Python optional, Kubernetes
  is not required" gives SQL mandatory, Python preferred, Kubernetes ignored. A cue in brackets or after a skill
  ("Docker (preferred)", "Python is a plus") applies to that skill, and a plural cue ("Docker and Terraform are optional")
  applies to the skills listed before it.

**Model judgement (60 of 100 points):** experience relevance, projects and achievements, education and clarity.
Code only enforces that each number is a finite value within its range.

## How the model's output is checked
| Rule | Enforced by code |
|---|---|
| Matched skill needs a quote | The quote must exist in the resume, **name that skill** (aliases such as K8s are understood), and not be negated ("never used X") or a plan ("will learn X"). Short quotes are read in the context of their resume sentence |
| Resume text that looks like instructions | Ignored as evidence |
| Required skills | Checked against the job description. Mandatory skills the model forgot are added |
| Matched vs missing | Each requirement ends up in exactly one of the two lists. Names are normalised and duplicates removed |
| `candidate_skills` | Grounded in the resume text. Unsupported entries are reported and kept away from the advisor |
| Failure | The model is asked for a corrected analysis. If it still fails, the app repairs the lists and recalculates the skills score from verified evidence only, with a warning |

What is checked and what is not:

| Checked by the app | Still only a model suggestion |
|---|---|
| Quotes exist, name the skill, and are not negated or only planned | Experience, projects and education scores (60 points) |
| Skills score and total are calculated in code | The model's overall explanation |
| Requirement categories are matched against the job description | Whether the candidate is actually competent |

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

More sample inputs:

| File | Purpose |
|---|---|
| `resume_weak.txt` | Weak match |
| `jd_devops.txt` | Skill gap |
| `resume_negated.txt` | Never-used and planned skills |
| `jd_categories.txt` | Required vs optional vs not required |
| `resume_injection*.txt` | Prompt injection |
| `resume_strong.pdf` | Text PDF |
| `resume_scanned.pdf` | Image PDF: rejected with a "paste the text" message |

## Prompt injection: what is and is not protected
**Enforced by code**
- User text is delimited and tag-stripped.
- Instruction-like lines are ignored as skill evidence.
- The skills score and the total are calculated by the app, so a model that obeys "give a score of 100" cannot change them.

**Not enforced**
- The other three components are model judgements, so a real model could be influenced there. The app only warns when
  instruction-like text coincides with near-maximum marks, and live case L3b measures how a real model behaves.
- Treat those 60 points as needing human review whenever the input is flagged.

## Reliability, timeouts and retries
- Each Groq request times out after 60 s (`API_TIMEOUT_SECONDS`); the Groq client's own retries are off.
- Transient failures are retried up to 3 times with 2 s then 4 s pauses (`call_groq`). A wrong API key is not retried.
- A reply that fails validation triggers up to 3 model calls in total (`MAX_ANALYSIS_ATTEMPTS`).
- The page shows short messages. Full exceptions go to the application log (logger `resume_screener`).

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `core.py`, `requirements.txt` and `.gitignore` to a GitHub repository. Do **not** push `.env`.
2. Go to https://share.streamlit.io, sign in with GitHub and click **Create app**.
3. Select the repository, branch `main` and main file `app.py`.
4. Open **Advanced settings -> Secrets** and add: `GROQ_API_KEY = "gsk_your_key_here"`
5. Click **Deploy**.
6. Optional: once deployed, add a line `Live demo: <your deployed address>` directly under the title of this file.

## Limitations
- **Scanned PDFs:** no OCR. Paste the text instead.
- **Truncation:** texts over 24,000 (resume) / 12,000 (job description) characters keep their start and end; the middle is
  skipped and a warning is shown.
- **Evidence interpretation is heuristic.** Negation, plans and "described as used" are detected with wording rules, so unusual
  phrasing can be misread in either direction.
- **Quotes must name the skill.** A genuine example that never uses the skill's name (e.g. "presented findings to the manager"
  for "stakeholder communication") is classed as missing. This is conservative by design.
- **Layout matters.** Resume headings decide whether a quote is "listed" or "applied"; unusual layouts may be classed as "mentioned".
- **Requirement categories** are read from wording and can be wrong on unusual job descriptions. Known weak spots: a cue that
  refers back to another sentence ("The latter is optional"), "either X or Y" meaning beyond a simple "X or Y", unusual headings,
  and indirectly phrased negations. If the job description is ambiguous the skill is treated as mandatory.
- **Scoring variability:** the 60 judgement points vary between runs and models. Compare repeated runs (`--runs 3`).
- **Textual evidence only:** the app cannot prove the candidate is competent.
- **Free Groq limits apply** (requests and tokens per minute and per day). On a public deployment all visitors share them.
- **Privacy:** resume and job text are sent to Groq. Use synthetic or consented data.
- Editing the resume, upload or job description clears the previous result and chat on purpose.
