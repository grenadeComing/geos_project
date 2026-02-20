import json
import re
from pathlib import Path
from typing import Any

import gradio as gr


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "build_dataset.jsonl"
CHOICES_TEMPLATE = "12\n224\n29\n13"
CHOICES_TEMPLATE_WITH_LETTERS = "A. 12\nB. 224\nC. 29\nD. 13"


def _norm(text: str) -> str:
    return (text or "").strip()


def _validate(stem: str, a: str, b: str, c: str, d: str, answer: str) -> list[str]:
    errors: list[str] = []
    if not stem:
        errors.append("Question is required.")
    choices = [a, b, c, d]
    if any(not x for x in choices):
        errors.append("All 4 choices are required.")
    if len(set(choices)) != 4:
        errors.append("Choices must be different.")
    if answer not in {"A", "B", "C", "D"}:
        errors.append("Answer must be A/B/C/D.")
    return errors


def _build_question(stem: str, a: str, b: str, c: str, d: str) -> str:
    return (
        f"{stem}\n\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}"
    )


def _parse_choices_blob(blob: str) -> tuple[str, str, str, str] | None:
    text = (blob or "").strip()
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return None

    cleaned: list[str] = []
    for ln in lines[:4]:
        # Accept common prefixes like "A.)", "A)", "A.", "A:" and strip them.
        stripped = re.sub(r"^[A-Da-d]\s*[\)\.\:]\s*[\)\.]?\s*", "", ln).strip()
        cleaned.append(stripped if stripped else ln)
    if len(cleaned) != 4 or any(not c for c in cleaned):
        return None
    return cleaned[0], cleaned[1], cleaned[2], cleaned[3]


def submit_entry(
    stem: str,
    choices_blob: str,
    answer: str,
) -> tuple[str, str, str, str, str]:
    stem = _norm(stem)
    answer = _norm(answer).upper()
    parsed = _parse_choices_blob(choices_blob)
    if parsed is None:
        return (
            "Submit failed:\n- Choices must be 4 lines (A/B/C/D).",
            "",
            stem,
            choices_blob,
            answer or "A",
        )
    a, b, c, d = (_norm(parsed[0]), _norm(parsed[1]), _norm(parsed[2]), _norm(parsed[3]))

    errors = _validate(stem, a, b, c, d, answer)
    if errors:
        return (
            "Submit failed:\n- " + "\n- ".join(errors),
            "",
            stem,
            choices_blob,
            answer or "A",
        )

    row = {"question": _build_question(stem, a, b, c, d), "answer": answer}
    with DATASET_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = count_rows()
    return (
        f"Saved to build_dataset.jsonl as row {total}.",
        json.dumps(row, ensure_ascii=False, indent=2),
        "",
        "",
        "A",
    )


def read_rows() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # keep review usable even if one bad line exists
                rows.append({"question": f"[INVALID JSON] {line}", "answer": "?"})
    return rows


def count_rows() -> int:
    return len(read_rows())


def review_rows(limit: int = 20) -> str:
    rows = read_rows()
    if not rows:
        return "No rows in build_dataset.jsonl"
    start = max(0, len(rows) - limit)
    lines: list[str] = []
    for i in range(start, len(rows)):
        row = rows[i]
        row_num = i + 1  # 1-based row number
        ans = row.get("answer", "")
        q = row.get("question", "")
        q1 = q.splitlines()[0] if isinstance(q, str) and q else ""
        lines.append(f"{row_num}. [{ans}] {q1[:120]}")
    return "\n".join(lines)


def delete_row(row_number: int) -> str:
    rows = read_rows()
    if row_number < 1 or row_number > len(rows):
        return "Row number out of range."
    del rows[row_number - 1]
    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return f"Deleted row {row_number}. Total rows now: {len(rows)}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Simple Dataset UI") as app:
        gr.Markdown("## Simple Dataset UI\nSubmit appends directly to `build_dataset.jsonl`.")

        with gr.Tab("Submit"):
            gr.Markdown(
                "**How to paste choices (easy mode)**\n"
                "- Paste 4 options, one per line.\n"
                "- You do **not** need to type `A/B/C/D`.\n"
                "- The app will automatically format saved question as `A.`, `B.`, `C.`, `D.`\n\n"
                "**Example without letters (recommended):**\n"
                f"```text\n{CHOICES_TEMPLATE}\n```\n"
                "**Also accepted:**\n"
                f"```text\n{CHOICES_TEMPLATE_WITH_LETTERS}\n```"
            )
            stem = gr.Textbox(label="Question", lines=8)
            choices_blob = gr.Textbox(
                label="Choices (4 lines, letters optional)",
                lines=6,
                placeholder=CHOICES_TEMPLATE,
            )
            answer = gr.Dropdown(choices=["A", "B", "C", "D"], label="Correct Answer", value="A")
            submit_btn = gr.Button("Submit", variant="primary")
            submit_status = gr.Textbox(label="Status", interactive=False)
            submit_preview = gr.Code(label="Saved Row", language="json")

            submit_btn.click(
                fn=submit_entry,
                inputs=[stem, choices_blob, answer],
                outputs=[submit_status, submit_preview, stem, choices_blob, answer],
            )

        with gr.Tab("Review"):
            refresh_btn = gr.Button("Refresh")
            recent = gr.Textbox(label="Recent Rows", lines=18, interactive=False)
            row_number = gr.Number(label="Row Number (1-based)", value=1, precision=0)
            delete_btn = gr.Button("Delete Row", variant="stop")
            review_status = gr.Textbox(label="Review Status", interactive=False)

            refresh_btn.click(fn=review_rows, inputs=[], outputs=[recent])
            delete_btn.click(fn=delete_row, inputs=[row_number], outputs=[review_status])

    return app


if __name__ == "__main__":
    build_ui().launch()
