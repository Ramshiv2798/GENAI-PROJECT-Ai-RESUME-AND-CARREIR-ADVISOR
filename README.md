# AI Resume Screener and Career Advisor

Reads a resume (PDF/TXT/pasted text) and a job description, then returns an
evidence-based **fit score out of 100**, matched and missing skills, and **three
suggestions** grounded in the candidate's real experience. A streamed chat
advisor answers follow-up questions.

The UI is built with **Gradio** (not Streamlit). Start it with `python app.py`.

## Project structure
```
app.py                 Gradio UI (streaming chat, state handling)
core.py                Prompts, PDF reading, Groq calls, strict validation
tests/test_core.py     Automated tests (Groq API mocked)
TESTING.md             Documented test cases and live-demo checklist
samples/               Strong, weak and prompt-injection sample resumes + a JD
.env.example           Dummy config (copy to .env)
.gitignore             Keeps .env and virtual environments out of Git
```

## Setup (Windows)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Edit `.env` and add your own key from https://console.groq.com/keys:
```
GROQ_API_KEY=your_key_here
MODEL_NAME=openai/gpt-oss-120b
```
Run:
```
python app.py
```
Open the URL printed in the terminal (normally http://127.0.0.1:7860).

Mac/Linux: use `python3 -m venv venv` and `source venv/bin/activate`.

## How it works
1. Resume (PDF/TXT/pasted) is read with `PyPDFLoader` and cleaned.
2. Long documents are split into chunks; the **start and end** are kept and the
   user sees a warning that the middle was omitted.
3. A role-based ("senior technical recruiter"), few-shot prompt asks for JSON:
   skills, matched skills **with a verbatim resume quote**, missing skills, four
   score components, reasoning and exactly three suggestions.
4. Python strictly validates the JSON (object, required fields, types, finite
   in-range numbers, three distinct suggestions) and retries with feedback if
   invalid.
5. Python **calculates the total score** by adding the four components
   (skills 40, experience 25, projects 20, education/clarity 15).
6. Every matched-skill quote is checked against the resume text; skills whose
   evidence is not found move to "missing" with a warning.
7. The chat advisor receives the resume, job description, analysis and the last
   10 messages, and streams its answer.

## Safety and correctness features
- Resume and job-description text is wrapped in delimiters and declared
  untrusted data; text that tries to override the instructions is flagged,
  cannot set the score, and cannot close the delimiter tags.
- If an analysis fails, **all state is cleared** (resume, analysis, chat) so the
  advisor never answers using a previous candidate.
- Suggestions must not tell candidates to claim skills, tools or numbers they
  have not actually achieved.
- Clear messages for missing inputs, scanned PDFs, corrupted files, API key
  errors, rate limits and outages.

## Security
- Never commit `.env`. It is listed in `.gitignore`; only `.env.example` (dummy
  value) belongs in the repository.
- If a key was ever committed or shared: revoke it at
  https://console.groq.com/keys, create a new one, remove the file from Git
  (`git rm --cached .env`) and purge it from history (see below).

Remove tracked secrets and the old virtual environment, then purge history:
```
git rm -r --cached .env BOT
git add .gitignore .env.example
git commit -m "Remove secrets and virtual environment from tracking"
pip install git-filter-repo
git filter-repo --path .env --path BOT --invert-paths --force
git remote add origin <your-repo-url>      # filter-repo removes remotes
git push --force --all
```
(Revoking the key is what actually protects you; history cleaning is extra.)

## Tests
```
pip install -r requirements-dev.txt
pytest -v
```
See `TESTING.md` for the documented test cases.

## Deployment (Hugging Face Spaces)
1. Create a new Space, choose the **Gradio** SDK.
2. Upload `app.py`, `core.py`, `requirements.txt` (not `.env`).
3. Space Settings -> Secrets: add `GROQ_API_KEY` (and optionally `MODEL_NAME`).
4. Put your Space URL here: `<ADD YOUR LIVE LINK>`.
