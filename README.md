# AI Resume Screener and Career Advisor

Reads a resume (PDF/text) and a job description, then returns:
- a fit score out of 100
- matched and missing skills
- three concrete suggestions to improve the resume
- a chat interface for follow-up career questions

## Setup
1. Install Python 3.10+.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   pip install -r requirements.txt
   ```
3. Get a free API key at https://console.groq.com/keys
4. Copy `.env.example` to `.env` and paste your key:
   ```
   GROQ_API_KEY=gsk_xxxxxxxx
   ```
5. Run: `python app.py` and open http://127.0.0.1:7860

## How it works
Resume (PDF/TXT/pasted) -> PyPDFLoader -> chunking (long docs) ->
role-based + few-shot prompt -> Groq LLM (JSON mode) -> validation ->
score, skills, suggestions -> chat advisor with the same context.

## Course concepts used
- Role-based prompting ("act as a senior technical recruiter")
- Few-shot example to fix the scoring format and rubric
- Structured JSON output (validated, clamped, retried if malformed)
- PDF reading and chunking (PyPDFLoader + RecursiveCharacterTextSplitter)
- API integration with error handling (retries, auth and rate-limit messages)

## Deploy on Hugging Face Spaces
Add `GROQ_API_KEY` under Settings -> Secrets and upload these files.
