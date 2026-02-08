import os
from tools.base_tool import BaseTool
from agent import run_agent

import gradio as gr


GREETING = "PHREEQC Agent ready. Paste a PHREEQC input or ask what to run."


def chat_fn(user_text, history):
    # history is ignored because we manage messages in state
    messages = []

    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})

    if user_text.strip().lower() in ["quit", "exit"]:
        return "Session reset."

    messages.append({"role": "user", "content": user_text})
    reply, _ = run_agent(messages)

    return reply


def build_ui():
    BaseTool.allowed_root = os.path.join(os.getcwd(), "work_space")

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="PHREEQC Agent",
        description=GREETING
    )

    return demo


if __name__ == "__main__":
    build_ui().launch()
