import os
import streamlit as st
DEFAULT_REPORT = "Upload a resume and job description, then click Analyze."
# Make Streamlit secrets visible to core.py (which reads os.environ). This must happen
# BEFORE `import core`, because core reads MODEL_NAME when it is imported.
# Locally, the .env file is used instead.
try:
    for _name in ("GROQ_API_KEY", "MODEL_NAME"):
        if _name in st.secrets:
            os.environ.setdefault(_name, str(st.secrets[_name]))
except Exception:
    pass  # no secrets file present
import core  # noqa: E402
st.set_page_config(page_title="AI Resume Screener and Career Advisor", layout="wide")
# ---------- session state ----------
def init_state():
    defaults = {
        "state": None,            # analysis state passed to the advisor
        "report": DEFAULT_REPORT,
        "result": None,           # raw JSON output
        "messages": [],           # chat history: [{"role": ..., "content": ...}]
        "uploader_key": 0,        # bump to reset the file uploader
        "resume_text": "",
        "job_desc": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
init_state()

# ---------- callbacks ----------
def reset_analysis():
    """Forget the previous analysis, report and chat (but keep the user's typed inputs)."""
    st.session_state.state = None
    st.session_state.report = DEFAULT_REPORT
    st.session_state.result = None
    st.session_state.messages = []

def clear_all():
    reset_analysis()
    st.session_state.resume_text = ""
    st.session_state.job_desc = ""
    st.session_state.uploader_key += 1  # new key -> empty uploader


def analyze(uploaded_file, pasted_text, job_description):
    # Step 1: wipe the previous candidate FIRST. From here on, no failure of any kind
    # (upload, temp file, PDF, API, unexpected exception) can leave old data behind.
    reset_analysis()

    try:
        name = data = None
        if uploaded_file is not None:
            name = uploaded_file.name
            data = bytes(uploaded_file.getbuffer())
        state, report, result = core.analyze_upload(name, data, pasted_text, job_description)
    except Exception as error:
        state, report, result = (
            None,
            f"**Error:** The analysis could not be completed ({error}). "
            "Please upload the file again or paste the resume text.",
            None,
        )

    st.session_state.state = state
    st.session_state.report = report
    st.session_state.result = result
    st.session_state.messages = []


# ---------- layout ----------
st.title("AI Resume Screener and Career Advisor")
st.markdown(
    "Upload a resume, paste a job description, and get an evidence-based fit score, "
    "missing skills, and three improvement suggestions."
)

left, right = st.columns(2)

with left:
    resume_file = st.file_uploader(
        "Upload resume (PDF or TXT)",
        type=["pdf", "txt"],
        key=f"resume_file_{st.session_state.uploader_key}",
    )
    st.text_area(
        "...or paste resume text",
        height=150,
        key="resume_text",
        placeholder="Paste your resume here if you are not uploading a file.",
    )
    st.text_area(
        "Job description",
        height=250,
        key="job_desc",
        placeholder="Paste the full job description here.",
    )

    analyze_clicked = st.button("Analyze", type="primary")
    st.button("Clear all", on_click=clear_all)

if analyze_clicked:
    with st.spinner("Analyzing..."):
        analyze(resume_file, st.session_state.resume_text, st.session_state.job_desc)

with right:
    st.markdown(st.session_state.report)
    with st.expander("Raw JSON output"):
        if st.session_state.result is not None:
            st.json(st.session_state.result)

# ---------- career advisor chat ----------
st.markdown("## Career Advisor Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a follow-up question, e.g. How should I rewrite my project section?")

if prompt and prompt.strip():
    message = prompt.strip()
    history = list(st.session_state.messages)  # history before this turn

    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        if not st.session_state.state:
            answer = "Please analyze a resume and job description first."
            st.markdown(answer)
        else:
            placeholder = st.empty()
            answer = ""
            try:
                # core.chat_stream yields the cumulative partial answer
                for partial in core.chat_stream(message, history, st.session_state.state):
                    answer = partial
                    placeholder.markdown(answer)
            except Exception as error:
                answer = f"Sorry, I could not answer that. {error}"
                placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
