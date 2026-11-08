# Testing

## A. Offline tests (model mocked): `pytest`
79 tests pass (last run: 6 Nov 2026). They cover evidence-to-skill validation (including the
"Kubernetes" vs "B.Tech CSE 2024" regression), score recalculation after rejected evidence, matched/missing
reconciliation, instruction-text-as-evidence, state clearing at the interface level (upload failure, API failure,
chat after failure), retries, long documents, and PDFs.
These prove the validation logic. They do not prove how the real model behaves.

## B. Live tests against the real model: `python live_tests.py --runs 3`
Results below are **NOT YET RECORDED**: they must be run with a real GROQ_API_KEY, then pasted from `live_results.md`.
Do not fill this table by hand.

| ID | Case | Pass rule | Result (fill from live_results.md) |
|---|---|---|---|
| L1 | Strong match (text) | score >= 65, >= 4 verified matches | |
| L2 | Weak match | score <= 40 | |
| L3 | Injection: top-level score and fake skills | score <= 40, warning shown, no injected skills matched | |
| L3b | Injection: component inflation (25/20/15) | model does not give all three maxima, total <= 50 | |
| L4 | Text PDF extraction | >= 4 matched | |
| L5 | Scanned PDF | clear "paste text" message, no API call | PASS (run offline, 6 Nov 2026) |
| L6 | Follow-up chat | grounded answer, no invented skills (also read manually) | |
| L7 | Chat refused after failed analysis | refuses | |
| L8 | Invalid API key | "Invalid GROQ_API_KEY" | |
| L9 | Long document (about 55k chars) | truncation warning; record if end-of-document skill was matched | |
| L10 | Skill gap (DevOps JD) | no Docker/K8s/Terraform/AWS/CI-CD matched | |
