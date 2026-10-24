import os
import json
import re
import time
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ============================================================
# 1. Configuration
# ============================================================
st.set_page_config(page_title="AI Resume Screener and Career Advisor", layout="wide")
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. Create a .env file containing "
        "GROQ_API_KEY=your_key (get one free at https://console.groq.com)."
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MAX_CONTEXT_CHARS = 12000
# ============================================================
# 2. Prompts
# ============================================================
ANALYSIS_SYSTEM_PROMPT = """
Act as a senior technical recruiter with 15 years of experience screening
candidates. Compare the RESUME against the JOB DESCRIPTION objectively.

Return ONLY a valid JSON object with exactly these keys:
{
  "candidate_skills": [string],        // skills found in the resume
  "required_skills": [string],         // skills required by the job description
  "matched_skills": [string],          // required skills the resume clearly shows
  "missing_skills": [string],          // required skills NOT shown in the resume
  "fit_score": integer,                // 0-100
  "score_reasoning": string,           // 1-2 sentences justifying the score
  "suggestions": [string, string, string]  // EXACTLY three concrete resume improvements
}

Scoring rubric (use it every time so scores stay consistent):
- 40 points: required skills coverage
- 25 points: relevant experience and seniority
- 20 points: projects / achievements relevant to the role
- 15 points: education, certifications, resume clarity

Rules:
- Use only evidence present in the resume. Never invent skills.
- Suggestions must be specific and actionable, not generic advice.
- Output JSON only. No markdown, no commentary.
""".strip()

FEW_SHOT_USER = """
RESUME:
Priya Nair. B.Tech CSE 2024. Skills: Python, SQL, Pandas, Excel.
Project: Sales dashboard using Pandas and Matplotlib. Internship: 3 months
as data analyst intern, cleaned datasets and built weekly reports.

JOB DESCRIPTION:
Data Analyst. Required: Python, SQL, Tableau, statistics, A/B testing,
stakeholder communication. 0-2 years experience.
""".strip()

FEW_SHOT_ASSISTANT = json.dumps(
    {
        "candidate_skills": ["Python", "SQL", "Pandas", "Excel", "Matplotlib"],
        "required_skills": [
            "Python", "SQL", "Tableau", "Statistics",
            "A/B Testing", "Stakeholder Communication",
        ],
        "matched_skills": ["Python", "SQL"],
        "missing_skills": ["Tableau", "Statistics", "A/B Testing", "Stakeholder Communication"],
        "fit_score": 62,
        "score_reasoning": (
            "Strong match on Python and SQL with relevant analyst internship "
            "experience, but several required analytics skills are not shown."
        ),
        "suggestions": [
            "Add a Tableau or Power BI dashboard project and link it on the resume.",
            "Quantify the internship impact, e.g. 'automated weekly reports, saving 4 hours per week'.",
            "Add a statistics / A/B testing line (course or mini-project) to match the job requirements.",
        ],
    },
    indent=2,
)

CHAT_SYSTEM_PROMPT = """
Act as a friendly, honest career advisor and recruiter. You have the candidate's
resume, the job description, and an earlier fit analysis. Answer the user's
questions using that information. Be specific and practical. If the question
needs information that is not in the resume or job description, say so instead
of guessing. Keep answers concise and student-friendly.
""".strip()
# ============================================================
# 3. Reading and chunking documents
# ============================================================
def read_resume(uploaded_file, pasted_text):
    """Read resume text from a Streamlit uploaded file or pasted text."""
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            try:
                pages = PyPDFLoader(tmp_path).load()
                text = "\n".join(p.page_content for p in pages)
            finally:
                os.remove(tmp_path)
        elif suffix in (".txt", ".md"):
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        else:
            raise ValueError("Unsupported file type. Upload a PDF or TXT file.")

        if text.strip():
            return text
        raise ValueError(
            "No text could be read from the file. It may be a scanned image PDF. "
            "Try pasting the resume text instead."
        )

    if pasted_text and pasted_text.strip():
        return pasted_text

    raise ValueError("Please upload a resume or paste the resume text.")


def chunk_and_limit(text, max_chars=MAX_CONTEXT_CHARS):
    """Split long text into chunks and keep them in order until the size limit."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    kept, total = [], 0
    for chunk in splitter.split_text(text):
        if total + len(chunk) > max_chars:
            break
        kept.append(chunk)
        total += len(chunk)
    return "\n".join(kept)
# ============================================================
# 4. Groq API call with error handling and retries
# ============================================================
def call_groq(messages, json_mode=False, max_tokens=2000, temperature=0.2, attempts=3):
    """Call Groq, retrying temporary failures."""
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            kwargs = dict(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            if MODEL_NAME.startswith("qwen/qwen3"):
                kwargs["extra_body"] = {"reasoning_effort": "none"}
            elif MODEL_NAME.startswith("openai/gpt-oss"):
                kwargs["extra_body"] = {"reasoning_effort": "low"}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()

            if not content:
                raise ValueError("The model returned an empty response.")
            return content

        except Exception as error:
            last_error = error
            name = type(error).__name__
            message = str(error)

            if name == "AuthenticationError" or "invalid_api_key" in message:
                raise RuntimeError("Invalid GROQ_API_KEY. Check your .env file.") from error
            if name == "BadRequestError" and json_mode:
                json_mode = False
                continue
            if name == "BadRequestError":
                raise RuntimeError(f"The request was rejected by Groq: {message}") from error

            if attempt < attempts:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"The Groq service is unavailable or rate limited. Please try again.\n"
        f"Details: {last_error}"
    )


def parse_analysis(raw):
    """Parse and validate the model's JSON so the UI never breaks."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("The model did not return valid JSON.")
        data = json.loads(match.group(0))

    def as_list(value):
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    try:
        score = int(round(float(data.get("fit_score", 0))))
    except (TypeError, ValueError):
        score = 0

    suggestions = as_list(data.get("suggestions"))[:3]
    while len(suggestions) < 3:
        suggestions.append("Tailor your resume summary and bullet points to the job description keywords.")

    return {
        "candidate_skills": as_list(data.get("candidate_skills")),
        "required_skills": as_list(data.get("required_skills")),
        "matched_skills": as_list(data.get("matched_skills")),
        "missing_skills": as_list(data.get("missing_skills")),
        "fit_score": max(0, min(100, score)),
        "score_reasoning": str(data.get("score_reasoning", "")).strip(),
        "suggestions": suggestions,
    }
# ============================================================
# 5. Core feature: analyze resume vs job description
# ============================================================
def format_report(result):
    score = result["fit_score"]
    if score >= 75:
        verdict = "Strong fit"
    elif score >= 50:
        verdict = "Moderate fit"
    else:
        verdict = "Weak fit"

    def bullets(items):
        return "\n".join(f"- {item}" for item in items) if items else "- None found"

    suggestions = "\n".join(f"{i}. {s}" for i, s in enumerate(result["suggestions"], 1))

    return f"""## Fit Score: {score}/100 ({verdict})

{result['score_reasoning']}

### Matched Skills
{bullets(result['matched_skills'])}

### Missing Skills
{bullets(result['missing_skills'])}

### 3 Suggestions to Improve Your Resume
{suggestions}
"""
def analyze_resume(uploaded_file, pasted_text, job_description):
    if not job_description or not job_description.strip():
        raise ValueError("Please enter the job description.")

    resume_text = chunk_and_limit(read_resume(uploaded_file, pasted_text))
    jd_text = chunk_and_limit(job_description, max_chars=6000)
    user_prompt = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd_text}"

    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": user_prompt},
    ]

    result = None
    for _ in range(2):
        raw = call_groq(messages, json_mode=True)
        try:
            result = parse_analysis(raw)
            break
        except (ValueError, json.JSONDecodeError):
            continue

    if result is None:
        raise RuntimeError("The model returned invalid JSON twice. Please try again.")

    state = {"resume": resume_text, "jd": jd_text, "analysis": result}
    return state, format_report(result), result
# ============================================================
# 6. Chat (career advisor)
# ============================================================
def chat_reply(message, history, state):
    context = (
        f"RESUME:\n{state['resume']}\n\n"
        f"JOB DESCRIPTION:\n{state['jd']}\n\n"
        f"FIT ANALYSIS (JSON):\n{json.dumps(state['analysis'], indent=2)}"
    )
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT + "\n\n" + context}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    messages.append({"role": "user", "content": message})
    return call_groq(messages, max_tokens=1500, temperature=0.4)
# ============================================================
# 7. Streamlit interface
# ============================================================
st.session_state.setdefault("state", None)
st.session_state.setdefault("report", None)
st.session_state.setdefault("result_json", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("uploader_key", 0)

st.title("AI Resume Screener and Career Advisor")
st.caption(
    "Upload a resume, paste a job description, and get a fit score, "
    "missing skills, and three improvement suggestions."
)

left, right = st.columns(2)

with left:
    uploaded = st.file_uploader(
        "Upload resume (PDF or TXT)",
        type=["pdf", "txt"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    pasted = st.text_area(
        "...or paste resume text",
        height=150,
        key=f"pasted_{st.session_state.uploader_key}",
        placeholder="Paste your resume here if you are not uploading a file.",
    )
    jd = st.text_area(
        "Job description",
        height=250,
        key=f"jd_{st.session_state.uploader_key}",
        placeholder="Paste the full job description here.",
    )

    c1, c2 = st.columns(2)
    analyze_clicked = c1.button("Analyze", type="primary", use_container_width=True)
    clear_clicked = c2.button("Clear all", use_container_width=True)

if clear_clicked:
    for k in ("state", "report", "result_json"):
        st.session_state[k] = None
    st.session_state.history = []
    st.session_state.uploader_key += 1
    st.rerun()

if analyze_clicked:
    try:
        with st.spinner("Analyzing resume..."):
            state, report, result = analyze_resume(uploaded, pasted, jd)
        st.session_state.state = state
        st.session_state.report = report
        st.session_state.result_json = result
        st.session_state.history = []
    except Exception as error:
        st.session_state.report = None
        left.error(str(error))

with right:
    if st.session_state.report:
        st.markdown(st.session_state.report)
        with st.expander("Raw JSON output"):
            st.json(st.session_state.result_json)
    else:
        st.info("Upload a resume and job description, then click Analyze.")

st.divider()
st.subheader("Career Advisor Chat")

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
if prompt := st.chat_input("Example: How should I rewrite my project section for this role?"):
    if not st.session_state.state:
        st.warning("Please analyze a resume and job description first.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            with st.spinner("Thinking..."):
                reply = chat_reply(prompt, st.session_state.history, st.session_state.state)
        except Exception as error:
            reply = f"Sorry, I could not answer that. {error}"
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.history += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]
