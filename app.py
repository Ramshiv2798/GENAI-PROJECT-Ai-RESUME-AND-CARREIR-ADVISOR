"""Gradio UI for the AI Resume Screener and Career Advisor. Logic lives in core.py."""
import gradio as gr

import core


def analyze(file_path, pasted_text, job_description):
    # Any failure returns state=None and an empty chat, so the advisor can never
    # answer using a previous candidate's data.
    state, report, result = core.safe_analyze(file_path, pasted_text, job_description)
    return state, report, result, []


def chat(message, history, state):
    message = (message or "").strip()
    history = history or []

    if not message:
        yield history, ""
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    if not state:
        history[-1]["content"] = "Please analyze a resume and job description first."
        yield history, ""
        return

    yield history, ""  # clear the input box immediately
    try:
        for partial in core.chat_stream(message, history[:-2], state):
            history[-1]["content"] = partial
            yield history, ""
    except Exception as error:
        history[-1]["content"] = f"Sorry, I could not answer that. {error}"
        yield history, ""


def clear_all():
    return None, None, "", None, "Upload a resume and job description, then click Analyze.", None, [], ""


with gr.Blocks(title="AI Resume Screener and Career Advisor") as demo:
    session_state = gr.State(value=None)

    gr.Markdown(
        """
        # AI Resume Screener and Career Advisor
        Upload a resume, paste a job description, and get an evidence-based fit score,
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
                label="...or paste resume text", lines=6,
                placeholder="Paste your resume here if you are not uploading a file.",
            )
            job_desc = gr.Textbox(
                label="Job description", lines=10,
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
        analyze,
        inputs=[resume_file, resume_text, job_desc],
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
