import os
import json
import re
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. Configuration
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Create a .env file containing "
        "GROQ_API_KEY=your_key (get one free at https://console.groq.com)."
    )

client = Groq(api_key=GROQ_API_KEY)

MAX_CONTEXT_CHARS = 12000  # keeps prompts well inside the model's context window


# ============================================================
# 2. Prompts: role-based + few-shot + structured JSON
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

# Few-shot example: teaches the model the exact output format and scoring style.
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
def read_resume(file_path, pasted_text):
    """Read resume text from an uploaded PDF/TXT file or pasted text."""
    if file_path:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            pages = PyPDFLoader(str(path)).load()
            text = "\n".join(page.page_content for page in pages)
        elif suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
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
    """
    Split long text into overlapping chunks, then keep chunks in order until
    the size limit is reached. Prevents oversized prompts for long documents.
    """
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
    """Call Groq, retrying temporary failures. Raises RuntimeError with a clear message."""
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

            # Qwen3 normally "thinks" before answering. Turn that off so the
            # reply is clean JSON/text and faster.
            if MODEL_NAME.startswith("qwen/qwen3"):
                kwargs["extra_body"] = {"reasoning_effort": "none"}
            # GPT-OSS models always reason; "low" keeps answers fast and leaves
            # enough tokens for the visible reply.
            elif MODEL_NAME.startswith("openai/gpt-oss"):
                kwargs["extra_body"] = {"reasoning_effort": "low"}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            # Safety net: remove any <think>...</think> block Qwen may emit.
            content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()

            if not content:
                raise ValueError("The model returned an empty response.")
            return content

        except Exception as error:
            last_error = error
            name = type(error).__name__
            message = str(error)

            # Do not retry errors that retrying cannot fix.
            if name == "AuthenticationError" or "invalid_api_key" in message:
                raise RuntimeError("Invalid GROQ_API_KEY. Check your .env file.") from error
            if name == "BadRequestError" and json_mode:
                json_mode = False  # model may not support JSON mode; parser still extracts JSON
                continue
            if name == "BadRequestError":
                raise RuntimeError(f"The request was rejected by Groq: {message}") from error

            if attempt < attempts:
                time.sleep(2 * attempt)  # simple backoff (also helps with rate limits)

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


def analyze_resume(file_path, pasted_text, job_description, state):
    """Returns: state, report markdown, JSON dict, reset chat history."""
    try:
        if not job_description or not job_description.strip():
            raise ValueError("Please enter the job description.")

        resume_text = chunk_and_limit(read_resume(file_path, pasted_text))
        jd_text = chunk_and_limit(job_description, max_chars=6000)

        user_prompt = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd_text}"

        messages = [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": FEW_SHOT_USER},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
            {"role": "user", "content": user_prompt},
        ]

        result = None
        for _ in range(2):  # one retry if the JSON is malformed
            raw = call_groq(messages, json_mode=True)
            try:
                result = parse_analysis(raw)
                break
            except (ValueError, json.JSONDecodeError):
                continue

        if result is None:
            raise RuntimeError("The model returned invalid JSON twice. Please try again.")

        new_state = {"resume": resume_text, "jd": jd_text, "analysis": result}
        return new_state, format_report(result), result, []

    except Exception as error:
        return state, f"**Error:** {error}", None, []


# ============================================================
# 6. Chat interface (career advisor)
# ============================================================
def chat(message, history, state):
    message = (message or "").strip()
    history = history or []

    if not message:
        return history, ""

    if not state:
        reply = "Please analyze a resume and job description first, then ask me questions."
    else:
        context = (
            f"RESUME:\n{state['resume']}\n\n"
            f"JOB DESCRIPTION:\n{state['jd']}\n\n"
            f"FIT ANALYSIS (JSON):\n{json.dumps(state['analysis'], indent=2)}"
        )
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT + "\n\n" + context}
        ]
        messages += [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
        messages.append({"role": "user", "content": message})

        try:
            reply = call_groq(messages, max_tokens=1500, temperature=0.4)
        except Exception as error:
            reply = f"Sorry, I could not answer that. {error}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


def clear_all():
    return None, None, "", None, "Upload a resume and job description, then click Analyze.", None, [], ""


# ============================================================
# 7. Gradio interface
# ============================================================
with gr.Blocks(title="AI Resume Screener and Career Advisor") as demo:
    session_state = gr.State(value=None)

    gr.Markdown(
        """
        # AI Resume Screener and Career Advisor
        Upload a resume, paste a job description, and get a fit score,
        missing skills, and three improvement suggestions.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            resume_file = gr.File(
                label="Upload resume (PDF or TXT)",
                file_types=[".pdf", ".txt"],
                type="filepath",
            )
            resume_text = gr.Textbox(
                label="...or paste resume text",
                lines=6,
                placeholder="Paste your resume here if you are not uploading a file.",
            )
            job_desc = gr.Textbox(
                label="Job description",
                lines=10,
                placeholder="Paste the full job description here.",
            )
            analyze_button = gr.Button("Analyze", variant="primary")
            clear_button = gr.Button("Clear all")

        with gr.Column(scale=1):
            report = gr.Markdown("Upload a resume and job description, then click Analyze.")
            with gr.Accordion("Raw JSON output", open=False):
                json_output = gr.JSON()

    gr.Markdown("## Career Advisor Chat")
    chatbot = gr.Chatbot(type="messages", height=380, label="Ask follow-up questions")
    chat_box = gr.Textbox(
        label="Ask a question",
        placeholder="Example: How should I rewrite my project section for this role?",
    )
    ask_button = gr.Button("Ask")

    analyze_button.click(
        fn=analyze_resume,
        inputs=[resume_file, resume_text, job_desc, session_state],
        outputs=[session_state, report, json_output, chatbot],
    )
    ask_button.click(chat, [chat_box, chatbot, session_state], [chatbot, chat_box])
    chat_box.submit(chat, [chat_box, chatbot, session_state], [chatbot, chat_box])
    clear_button.click(
        clear_all,
        outputs=[resume_file, resume_text, job_desc, session_state, report, json_output, chatbot, chat_box],
    )

if __name__ == "__main__":
    demo.queue().launch()
