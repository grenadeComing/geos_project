"""
Shared helpers for the three runners:
  - oneshot.py         (one-shot baseline, no tools)
  - evaluate_custom.py (custom agent, 4 context modes)
  - evaluate_sdk.py    (Claude Agent SDK)

Design goals:
  1. Single source of truth for prompts, parsers, graders.
  2. Dual task support (MCQ + numeric fill-in-the-blank) via a `task_type` field
     on each dataset row. See `load_dataset` for the schema.
  3. Numeric fill grading supports relative tolerance for generic numeric
     rows and item-specific absolute tolerance for the NeurIPS FITB subset.

Dataset schema (phreeqc_bench_v2.jsonl):

    {
      "id":        int,          # optional; derived from line index if absent
      "question":  str,          # the prompt shown to the model
      "task_type": "mcq"|"fill",
      "gold":      str|float,    # "A"/"B"/"C"/"D" for MCQ; number for fill
      "tolerance": float         # optional, overrides default 1e-3 (numeric fill only)
    }

Fill-in-the-blank JSONL rows may also use the compact benchmark schema:

    {
      "preamble":          str,   # optional instruction prepended to direct one-shot prompts
      "question":          str,
      "answer_value":      str,   # numeric gold answer
      "answer_tolerance":  str    # absolute tolerance derived from nearest distractor
    }
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


# ── Tolerance for numeric fill-in-blank grading ─────────────────────────
DEFAULT_TOLERANCE = 1e-3        # default relative tolerance for generic numeric fill rows
DEFAULT_ABS_TOLERANCE = 1e-12   # absolute floor only for relative-tolerance rows near zero


# ── System prompts ──────────────────────────────────────────────────────

GENERIC_SYSTEM_PROMPT = "You are a geochemistry expert."

MCQ_SYSTEM_PROMPT_ONESHOT = (
    "You are a geochemistry expert. "
    "Answer the multiple-choice question. "
    "Your entire response must be EXACTLY <<< X >>> where X is A, B, C, or D. "
    "No explanation, no reasoning, no other text. ONLY <<< X >>>."
)

FILL_SYSTEM_PROMPT_ONESHOT = (
    "You are a geochemistry expert. "
    "Answer the question with a single numeric value. "
    "Your entire response must be EXACTLY <<< VALUE >>> where VALUE is a number "
    "(e.g. 1.23e-5, 0.00123, 42). "
    "Use plain numeric format; do NOT include units inside the <<< >>> markers. "
    "No explanation, no reasoning, no other text. ONLY <<< VALUE >>>."
)

MCQ_SYSTEM_PROMPT_COT = (
    "You are a geochemistry expert. "
    "Solve the multiple-choice question by reasoning step by step, but do not use tools. "
    "End your response with the final answer in the exact form <<< X >>>, where X is A, B, C, or D."
)

FILL_SYSTEM_PROMPT_COT = (
    "You are a geochemistry expert. "
    "Solve the question by reasoning step by step, but do not use tools. "
    "End your response with the final answer in the exact form <<< VALUE >>>, where VALUE is a number "
    "(e.g. 1.23e-5, 0.00123, 42). "
    "Do not include units inside the <<< >>> markers."
)

# Agent-variant prompts: appended to the base agent system prompt in the runner.
MCQ_FINAL_ANSWER_INSTRUCTION = (
    "Write your final answer to final_answer.txt with exact content: "
    "<<< X >>> followed by a newline, where X is A, B, C, or D."
)

FILL_FINAL_ANSWER_INSTRUCTION = (
    "Write your final answer to final_answer.txt with exact content: "
    "<<< VALUE >>> followed by a newline, where VALUE is a plain number "
    "(e.g. 1.23e-5, 0.00123, 42). "
    "Do NOT include units inside the <<< >>> markers."
)


# ── User-prompt builders ────────────────────────────────────────────────

def build_prompt_oneshot(row: dict[str, Any], *, include_preamble: bool = True) -> str:
    """Build the user prompt for the one-shot baseline (no tools)."""
    question = row["question"]
    preamble = str(row.get("preamble") or "").strip()
    if include_preamble and preamble:
        return f"{preamble}\n\n{question}"
    return question


def build_prompt_agent_mcq(row: dict[str, Any]) -> str:
    return (
        "Answer the multiple-choice question below.\n\n"
        f"Question:\n{row['question']}\n\n"
        "Use tools as needed for PHREEQC / quantitative work."
    )


def build_prompt_agent_fill(row: dict[str, Any]) -> str:
    return (
        "Answer the question below with a single numeric value.\n\n"
        f"Question:\n{row['question']}\n\n"
        "Use tools as needed for PHREEQC / quantitative work. "
        "The answer must be a single plain number with no units or explanation."
    )


def build_prompt(row: dict[str, Any], *, for_agent: bool) -> str:
    """Dispatch to the right builder based on task_type and whether this is agent vs one-shot."""
    tt = row["task_type"]
    if for_agent:
        return build_prompt_agent_mcq(row) if tt == "mcq" else build_prompt_agent_fill(row)
    return build_prompt_oneshot(row)


def system_prompt_oneshot(task_type: str, *, has_preamble: bool = False) -> str:
    if task_type == "fill" and has_preamble:
        return GENERIC_SYSTEM_PROMPT
    return MCQ_SYSTEM_PROMPT_ONESHOT if task_type == "mcq" else FILL_SYSTEM_PROMPT_ONESHOT


def system_prompt_cot(task_type: str) -> str:
    return MCQ_SYSTEM_PROMPT_COT if task_type == "mcq" else FILL_SYSTEM_PROMPT_COT


# ── Answer parsers ──────────────────────────────────────────────────────

_WRAPPED = re.compile(r"<<<\s*(.+?)\s*>>>", re.DOTALL)
_MCQ_CHOICE = re.compile(r"^([A-D])$", re.IGNORECASE)
_MCQ_CHOICE_LOOSE = re.compile(r"\b([A-D])\b")
_NUMERIC_TOKEN = re.compile(r"^[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?$")


def _coerce_float(value: Any) -> float | None:
    """Return a finite float from a plain numeric value or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if not isinstance(value, str):
        return None
    token = value.strip().split()[0].rstrip(",;") if value.strip() else ""
    if not _NUMERIC_TOKEN.match(token):
        return None
    out = float(token)
    return out if math.isfinite(out) else None


def parse_answer(text: str | None, task_type: str, *, strict: bool = False) -> str | float | None:
    """Extract the answer from model output / final_answer.txt content.

    For MCQ: returns an uppercase letter A/B/C/D, or None if not found.
    For fill: returns a float if the content starts with a plain numeric value,
      otherwise None.

    When `strict=True`, the input MUST be wrapped in `<<< ... >>>` markers;
    otherwise None is returned. This matches the original ACM-era parsers for
    agent/SDK final_answer.txt, which required the exact wrapper.

    When `strict=False` (default), parsing is permissive: if the wrapper is
      missing, we try loose MCQ letter detection (`\\b[A-D]\\b`) or parse a
      leading numeric value for fill rows.
    """
    if not text:
        return None
    m = _WRAPPED.search(text)
    if strict and m is None:
        return None
    inner = m.group(1).strip() if m else text.strip()

    if task_type == "mcq":
        m2 = _MCQ_CHOICE.match(inner)
        if m2:
            return m2.group(1).upper()
        if strict:
            # Original agent/SDK _parse_choice accepted only `<<<[A-D]>>>`.
            # Anything else (even `<<<Answer: B>>>`) was rejected.
            return None
        m3 = _MCQ_CHOICE_LOOSE.search(inner)
        return m3.group(1).upper() if m3 else None

    # Numeric fill: allow a trailing unit token for robustness, but require
    # the first token to be a finite plain number.
    if not inner:
        return None
    return _coerce_float(inner)


# ── Graders ─────────────────────────────────────────────────────────────

def grade(pred: str | float | None, gold: str | float, task_type: str,
          tolerance: float = DEFAULT_TOLERANCE,
          tolerance_mode: str = "relative") -> bool | None:
    """Return True/False if we can grade, None if prediction is unparseable."""
    if pred is None:
        return None

    if task_type == "mcq":
        if not isinstance(pred, str):
            return None
        return pred.strip().upper() == str(gold).strip().upper()

    gold_num = _coerce_float(gold)
    pred_num = _coerce_float(pred)
    if gold_num is None or pred_num is None:
        return None
    if tolerance_mode == "absolute":
        return abs(pred_num - gold_num) <= tolerance
    return math.isclose(pred_num, gold_num,
                        rel_tol=tolerance, abs_tol=DEFAULT_ABS_TOLERANCE)


# ── Dataset loading ─────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load benchmark JSONL. Rows must have question + task_type + gold.

    Backward-compat: rows that only have `answer` (as in the ACM-era dataset)
    are treated as MCQ with gold = answer.

    Compact fill-in-the-blank rows with `answer_value` are treated as
    task_type=fill. If `answer_tolerance` is present, it is interpreted as an
    absolute tolerance rather than the legacy relative tolerance.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question")
            task_type = row.get("task_type")
            gold = row.get("gold")
            tolerance_mode = str(row.get("tolerance_mode", "relative")).strip().lower()
            tolerance = row.get("tolerance", DEFAULT_TOLERANCE)
            if "answer_tolerance" in row:
                tolerance = row["answer_tolerance"]
                tolerance_mode = "absolute"
            if task_type is None and gold is None and "answer_value" in row:
                task_type = "fill"
                gold = row["answer_value"]
            # Legacy fallback: old rows have "answer" (MCQ only).
            if task_type is None and gold is None and "answer" in row:
                task_type = "mcq"
                gold = row["answer"]
            if not question or gold is None or task_type not in {"mcq", "fill"}:
                continue
            if tolerance_mode not in {"relative", "absolute"}:
                raise ValueError(
                    f"Unsupported tolerance_mode={tolerance_mode!r} "
                    f"at {path}:{line_idx + 1}"
                )
            tolerance_float = float(tolerance)
            if not math.isfinite(tolerance_float) or tolerance_float < 0:
                raise ValueError(f"Invalid tolerance={tolerance!r} at {path}:{line_idx + 1}")
            if task_type == "fill" and _coerce_float(gold) is None:
                raise ValueError(f"Numeric fill row has non-numeric gold at {path}:{line_idx + 1}")
            record = {
                # Internal zero-based row position. Runners use this for stable
                # workspace names regardless of whether the dataset has an id.
                "index": line_idx,
                # Public dataset id, preserved in result files.
                "id": int(row.get("id", line_idx + 1)),
                "question": str(question),
                "task_type": task_type,
                "gold": gold if not isinstance(gold, str) else gold.strip(),
                "tolerance": tolerance_float,
                "tolerance_mode": tolerance_mode,
            }
            if row.get("preamble"):
                record["preamble"] = str(row["preamble"]).strip()
            out.append(record)
    if not out:
        raise RuntimeError(f"No usable rows in {path}")
    return out


# ── Result helpers ──────────────────────────────────────────────────────

def grade_row(row: dict[str, Any], pred: str | float | None) -> dict[str, Any]:
    """Build a per-question result dict with consistent fields."""
    tol = float(row.get("tolerance", DEFAULT_TOLERANCE))
    tolerance_mode = str(row.get("tolerance_mode", "relative"))
    is_correct = grade(
        pred,
        row["gold"],
        row["task_type"],
        tolerance=tol,
        tolerance_mode=tolerance_mode,
    )
    return {
        "index": row.get("id", row["index"] + 1),
        "row_index": row["index"],
        "task_type": row["task_type"],
        "truth": row["gold"],
        "prediction": pred,
        "tolerance": tol if row["task_type"] == "fill" else None,
        "tolerance_mode": tolerance_mode if row["task_type"] == "fill" else None,
        "is_correct": is_correct,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate accuracy, plus per-task-type breakdown."""
    total = len(results)
    graded = [r for r in results if r.get("is_correct") is not None]
    correct = sum(1 for r in graded if r["is_correct"])
    errors = sum(1 for r in results if r.get("error"))
    ungraded = total - len(graded)

    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        tt = r.get("task_type", "unknown")
        slot = by_type.setdefault(
            tt,
            {"total": 0, "correct": 0, "graded": 0, "ungraded": 0, "errors": 0},
        )
        slot["total"] += 1
        if r.get("is_correct") is True:
            slot["correct"] += 1
        if r.get("is_correct") is not None:
            slot["graded"] += 1
        else:
            slot["ungraded"] += 1
        if r.get("error"):
            slot["errors"] += 1
    for slot in by_type.values():
        # Main accuracy treats unparseable / missing answers as wrong. This is
        # the right denominator for MCQ benchmark reporting.
        slot["accuracy"] = (slot["correct"] / slot["total"]) if slot["total"] else 0.0
        slot["graded_accuracy"] = (slot["correct"] / slot["graded"]) if slot["graded"] else 0.0

    return {
        "total_rows": total,
        "graded_rows": len(graded),
        "ungraded_rows": ungraded,
        "correct_rows": correct,
        "error_rows": errors,
        "accuracy": (correct / total) if total else 0.0,
        "graded_accuracy": (correct / len(graded)) if graded else 0.0,
        "by_task_type": by_type,
    }
