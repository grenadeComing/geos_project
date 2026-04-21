"""
Shared helpers for the three runners:
  - oneshot.py         (one-shot baseline, no tools)
  - evaluate_custom.py (custom agent, 4 context modes)
  - evaluate_sdk.py    (Claude Agent SDK)

Design goals:
  1. Single source of truth for prompts, parsers, graders.
  2. Dual task support (MCQ + fill-in-the-blank) via a `task_type` field
     on each dataset row. See `load_dataset` for the schema.
  3. Numeric fill grading uses tolerance = 1e-3 (relative + small absolute).

Dataset schema (phreeqc_bench_v2.jsonl):

    {
      "id":        int,          # optional; derived from line index if absent
      "question":  str,          # the prompt shown to the model
      "task_type": "mcq"|"fill",
      "gold":      str|float,    # "A"/"B"/"C"/"D" for MCQ; number or string for fill
      "tolerance": float         # optional, overrides default 1e-3 (numeric fill only)
    }
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


# ── Tolerance for numeric fill-in-blank grading ─────────────────────────
DEFAULT_TOLERANCE = 1e-3        # relative tolerance
DEFAULT_ABS_TOLERANCE = 1e-12   # absolute floor (protects values near 0)


# ── System prompts ──────────────────────────────────────────────────────

MCQ_SYSTEM_PROMPT_ONESHOT = (
    "You are a geochemistry expert. "
    "Answer the multiple-choice question. "
    "Your entire response must be EXACTLY <<< X >>> where X is A, B, C, or D. "
    "No explanation, no reasoning, no other text. ONLY <<< X >>>."
)

FILL_SYSTEM_PROMPT_ONESHOT = (
    "You are a geochemistry expert. "
    "Answer the question with a single value. "
    "Your entire response must be EXACTLY <<< VALUE >>> where VALUE is either "
    "a number (e.g. 1.23e-5, 0.00123, 42) or a chemical formula / short text answer "
    "(e.g. CaCO3). "
    "Use plain numeric format; do NOT include units inside the <<< >>> markers. "
    "No explanation, no reasoning, no other text. ONLY <<< VALUE >>>."
)

# Agent-variant prompts: appended to the base agent system prompt in the runner.
MCQ_FINAL_ANSWER_INSTRUCTION = (
    "Write your final answer to final_answer.txt with exact content: "
    "<<< X >>> followed by a newline, where X is A, B, C, or D."
)

FILL_FINAL_ANSWER_INSTRUCTION = (
    "Write your final answer to final_answer.txt with exact content: "
    "<<< VALUE >>> followed by a newline, where VALUE is either a plain number "
    "(e.g. 1.23e-5) or a short chemical formula / text answer (e.g. CaCO3). "
    "Do NOT include units inside the <<< >>> markers."
)


# ── User-prompt builders ────────────────────────────────────────────────

def build_prompt_oneshot(row: dict[str, Any]) -> str:
    """Build the user prompt for the one-shot baseline (no tools)."""
    return row["question"]


def build_prompt_agent_mcq(row: dict[str, Any]) -> str:
    return (
        "Answer the multiple-choice question below.\n\n"
        f"Question:\n{row['question']}\n\n"
        "Steps:\n"
        "1) Use tools as needed for PHREEQC / quantitative work.\n"
        "2) Your response should be in the form <<< X >>>, where X is either A, B, C, or D.\n"
        "3) Use write_file to create or overwrite final_answer.txt in the current workspace.\n"
        "   The file content must be exactly: <<< X >>> followed by a newline.\n"
    )


def build_prompt_agent_fill(row: dict[str, Any]) -> str:
    return (
        "Answer the question below with a single value.\n\n"
        f"Question:\n{row['question']}\n\n"
        "Steps:\n"
        "1) Use tools as needed for PHREEQC / quantitative work.\n"
        "2) Your answer must be a single value: either a plain number "
        "(e.g. 1.23e-5, 0.00123, 42) or a short chemical formula / text answer (e.g. CaCO3).\n"
        "3) Use write_file to create or overwrite final_answer.txt in the current workspace.\n"
        "   The file content must be exactly: <<< VALUE >>> followed by a newline.\n"
        "   Do NOT include units inside the <<< >>> markers.\n"
    )


def build_prompt(row: dict[str, Any], *, for_agent: bool) -> str:
    """Dispatch to the right builder based on task_type and whether this is agent vs one-shot."""
    tt = row["task_type"]
    if for_agent:
        return build_prompt_agent_mcq(row) if tt == "mcq" else build_prompt_agent_fill(row)
    return build_prompt_oneshot(row)


def system_prompt_oneshot(task_type: str) -> str:
    return MCQ_SYSTEM_PROMPT_ONESHOT if task_type == "mcq" else FILL_SYSTEM_PROMPT_ONESHOT


# ── Answer parsers ──────────────────────────────────────────────────────

_WRAPPED = re.compile(r"<<<\s*(.+?)\s*>>>", re.DOTALL)
_MCQ_CHOICE = re.compile(r"^([A-D])$", re.IGNORECASE)
_MCQ_CHOICE_LOOSE = re.compile(r"\b([A-D])\b")


def parse_answer(text: str | None, task_type: str, *, strict: bool = False) -> str | float | None:
    """Extract the answer from model output / final_answer.txt content.

    For MCQ: returns an uppercase letter A/B/C/D, or None if not found.
    For fill: returns a float if the content parses as a number, otherwise the
      stripped string (for formula-style answers like 'CaCO3'). Returns None
      if nothing extractable is found.

    When `strict=True`, the input MUST be wrapped in `<<< ... >>>` markers;
    otherwise None is returned. This matches the original ACM-era parsers for
    agent/SDK final_answer.txt, which required the exact wrapper.

    When `strict=False` (default), parsing is permissive: if the wrapper is
    missing, we try loose MCQ letter detection (`\\b[A-D]\\b`) or use the full
    text as the fill value. This matches the lenient behavior of the ACM
    one-shot baseline scripts.
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

    # Fill: try numeric first, then fall back to string.
    if not inner:
        return None
    # Strip a trailing unit-ish token (e.g. "1.23e-5 mol/L" → "1.23e-5").
    # We're conservative: only strip if the leading token parses as a number.
    tok = inner.split()[0].rstrip(",;")
    try:
        return float(tok)
    except ValueError:
        pass
    try:
        return float(inner)
    except ValueError:
        pass
    return inner


# ── Graders ─────────────────────────────────────────────────────────────

def _normalize_string(s: str) -> str:
    """Normalize a string answer for comparison: strip, casefold, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().casefold())


def grade(pred: str | float | None, gold: str | float, task_type: str,
          tolerance: float = DEFAULT_TOLERANCE) -> bool | None:
    """Return True/False if we can grade, None if prediction is unparseable."""
    if pred is None:
        return None

    if task_type == "mcq":
        if not isinstance(pred, str):
            return None
        return pred.strip().upper() == str(gold).strip().upper()

    # Fill grading.
    # If gold is numeric (or numeric-looking string), do numeric comparison.
    gold_num: float | None = None
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        gold_num = float(gold)
    elif isinstance(gold, str):
        try:
            gold_num = float(gold)
        except ValueError:
            gold_num = None

    if gold_num is not None:
        # Coerce pred to float if it's numeric or numeric-looking.
        if isinstance(pred, (int, float)) and not isinstance(pred, bool):
            pred_num: float | None = float(pred)
        elif isinstance(pred, str):
            try:
                pred_num = float(pred)
            except ValueError:
                return False
        else:
            return False
        if math.isnan(pred_num) or math.isnan(gold_num):
            return False
        return math.isclose(pred_num, gold_num,
                            rel_tol=tolerance, abs_tol=DEFAULT_ABS_TOLERANCE)

    # String fill: normalized equality.
    if not isinstance(pred, str):
        pred = str(pred)
    return _normalize_string(pred) == _normalize_string(str(gold))


# ── Dataset loading ─────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load phreeqc_bench_v2 JSONL. Rows must have question + task_type + gold.

    Backward-compat: rows that only have `answer` (as in the ACM-era dataset)
    are treated as MCQ with gold = answer.
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
            # Legacy fallback: old rows have "answer" (MCQ only).
            if task_type is None and gold is None and "answer" in row:
                task_type = "mcq"
                gold = row["answer"]
            if not question or gold is None or task_type not in {"mcq", "fill"}:
                continue
            record = {
                "index": int(row.get("id", line_idx)),
                "question": str(question),
                "task_type": task_type,
                "gold": gold if not isinstance(gold, str) else gold.strip(),
                "tolerance": float(row.get("tolerance", DEFAULT_TOLERANCE)),
            }
            out.append(record)
    if not out:
        raise RuntimeError(f"No usable rows in {path}")
    return out


# ── Result helpers ──────────────────────────────────────────────────────

def grade_row(row: dict[str, Any], pred: str | float | None) -> dict[str, Any]:
    """Build a per-question result dict with consistent fields."""
    tol = float(row.get("tolerance", DEFAULT_TOLERANCE))
    is_correct = grade(pred, row["gold"], row["task_type"], tolerance=tol)
    return {
        "index": row["index"] + 1,
        "task_type": row["task_type"],
        "truth": row["gold"],
        "prediction": pred,
        "tolerance": tol if row["task_type"] == "fill" else None,
        "is_correct": is_correct,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate accuracy, plus per-task-type breakdown."""
    total = len(results)
    graded = [r for r in results if r.get("is_correct") is not None]
    correct = sum(1 for r in graded if r["is_correct"])
    errors = sum(1 for r in results if r.get("error"))

    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        tt = r.get("task_type", "unknown")
        slot = by_type.setdefault(tt, {"total": 0, "correct": 0, "graded": 0, "errors": 0})
        slot["total"] += 1
        if r.get("is_correct") is True:
            slot["correct"] += 1
        if r.get("is_correct") is not None:
            slot["graded"] += 1
        if r.get("error"):
            slot["errors"] += 1
    for slot in by_type.values():
        slot["accuracy"] = (slot["correct"] / slot["graded"]) if slot["graded"] else 0.0

    return {
        "total_rows": total,
        "graded_rows": len(graded),
        "correct_rows": correct,
        "error_rows": errors,
        "accuracy": (correct / len(graded)) if graded else 0.0,
        "by_task_type": by_type,
    }
