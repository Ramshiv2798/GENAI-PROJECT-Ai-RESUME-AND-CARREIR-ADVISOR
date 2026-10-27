# Testing

## A. Automated tests (Groq API mocked)
Run `pytest -v`. Result at the time of writing: **27 passed**.

| ID | Scenario | Expected result | Test |
|---|---|---|---|
| TC01 | Strong match | Score equals sum of 4 components (74); matched skills with evidence; 3 suggestions | `test_tc01_*` |
| TC02 | Weak match | Low score (12) and "Weak fit" verdict | `test_tc02_*` |
| TC03 | Missing job description | Clear error, no state | `test_tc03_*` |
| TC04 | Missing resume | Clear error, no state | `test_tc04_*` |
| TC05 | Scanned/image-only PDF | Message explaining no text found, asks to paste text | `test_tc05_*` |
| TC06 | Malformed JSON | Retried with correction; after 3 failures a clear error | `test_tc06_*`, `test_tc06b_*` |
| TC07 | `{}`, JSON list, bad suggestions, missing/out-of-range components, infinite or non-numeric score | Rejected (ValueError), triggers retry | `test_tc07_*`, `test_tc07b_*` |
| TC08 | Very long resume | Start and end both kept; truncation warning shown | `test_tc08_*` |
| TC09 | API outage / invalid key / JSON mode unsupported | Friendly error; invalid key not retried; fallback without JSON mode | `test_tc09_*` |
| TC10 | Follow-up chat | Prompt contains resume, JD, analysis, last 10 turns; refuses with no analysis | `test_tc10_*` |
| TC11 | New analysis fails | State is `None` (no previous candidate data survives) | `test_tc11_*` |
| TC12 | Prompt injection in resume | Text delimited, flagged, fake closing tag removed, model-supplied `fit_score` ignored | `test_tc12_*` |
| TC13 | Quote not found in resume | Skill moved to missing with warning | `test_tc13_*` |
| TC14 | Unsupported file type | "Unsupported file type" message | `test_tc14_*` |

Limits of these tests: the model is mocked, so they prove the application's
validation and handling, not the model's real behavior.

## B. Live tests with the real Groq API (fill in yourself)
Use the files in `samples/`. Record the real output and a screenshot for each.

| ID | Input | What to check | Result | Pass? |
|---|---|---|---|---|
| L1 | `resume_strong.txt` + `jd_data_analyst.txt` | High score; matched skills have correct quotes | | |
| L2 | `resume_weak.txt` + `jd_data_analyst.txt` | Low score; missing skills listed | | |
| L3 | `resume_injection.txt` + `jd_data_analyst.txt` | Warning shown; score stays low; no "100" | | |
| L4 | A real text PDF resume | Skills extracted correctly | | |
| L5 | A scanned PDF resume | Helpful "paste text" message | | |
| L6 | Analyze L1, ask "What should I improve first?" | Streamed answer grounded in the resume | | |
| L7 | Analyze L1, then run an invalid analysis (empty JD) and ask a question | Chat says to analyze first (no old data) | | |
| L8 | Set a wrong key in `.env` | "Invalid GROQ_API_KEY" message | | |

## C. Demo evidence to attach
- [ ] Deployment link (Hugging Face Space) or recorded demo video
- [ ] Screenshots of L1-L8
