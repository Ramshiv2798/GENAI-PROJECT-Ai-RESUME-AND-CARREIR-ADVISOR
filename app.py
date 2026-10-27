import os
import tempfile
import streamlit as st
import core
DEFAULT_REPORT = "Upload a resume and job description, then click Analyze."

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
def clear_all():
    st.session_state.state = None
    st.session_state.report = DEFAULT_REPORT
    st.session_state.result = None
    st.session_state.messages = []
    st.session_state.resume_text = ""
    st.session_state.job_desc = ""
    st.session_state.uploader_key += 1  # new key -> empty uploader


def analyze(uploaded_file, pasted_text, job_description):
    # Any failure leaves state=None and an empty chat, so the advisor can never
    # answer using a previous candidate's data.
    tmp_path = None
    try:
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

        state, report, result = core.safe_analyze(tmp_path, pasted_text, job_description)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

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
