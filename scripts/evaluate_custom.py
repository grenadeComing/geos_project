"""
Custom agent evaluation runner. This is the primary script for comparing
context-management strategies at the tool-return boundary.

Supported context modes (via --context-mode):

  full     Full content of result.out returned inline (capped at MAX_RAW_CHARS,
           effectively unlimited for this benchmark).
  toc      Section-level table of contents returned; agent uses read_file
           with start_line/end_line to inspect sections. (Our TOC pattern.)
  summary  Question-conditioned LLM summary of result.out returned.
  rag      Embedding-based retrieval of top-k chunks from result.out,
           ranked by cosine similarity to the question.

Supports MCQ + fill-in-the-blank via dataset `task_type` field (see _common.py).

Usage:
  python scripts/evaluate_custom.py --model claude-opus-4-6 \\
      --context-mode toc --name custom_toc_opus46 --workers 4
  python scripts/evaluate_custom.py --model gpt-5.2 \\
      --context-mode summary --name custom_summary_gpt52 --workers 4
      # (summary uses the agent's own model by default; override with --summary-model)
  python scripts/evaluate_custom.py --model claude-sonnet-4-6 \\
      --context-mode rag --rag-k 3 --name custom_rag_sonnet46 --workers 4

Requires: pip install litellm openai
Requires: ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY as appropriate.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from _common import (  # noqa: E402
    build_prompt,
    grade_row,
    load_dataset,
    parse_answer,
    summarize_results,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


# ── Paths & constants ───────────────────────────────────────────────────

DEFAULT_DATASET = REPO_ROOT / "datasets" / "phreeqc_bench_v2.jsonl"
RESULT_ROOT = REPO_ROOT / "result" / "custom"

# Agent configs (reuse the original PHREEQC-agent system prompt file).
AGENT_CONFIGS_DIR = REPO_ROOT / "agent" / "configs"

# PHREEQC executable + database: kept at repo root so runners can share.
# Falls back to the ACM archive copy if not present at root.
PHREEQC_BIN = os.environ.get(
    "PHREEQC_BIN",
    "/Users/kezhang/Desktop/projects/geos/phreeqc-3.8.6-17100/src/phreeqc",
)
DEFAULT_DB = REPO_ROOT / "phreeqc.dat"
if not DEFAULT_DB.exists():
    DEFAULT_DB = REPO_ROOT / "archive_acm_2026" / "phreeqc.dat"
DB_REFERENCE = REPO_ROOT / "database_reference.txt"
if not DB_REFERENCE.exists():
    DB_REFERENCE = REPO_ROOT / "archive_acm_2026" / "database_reference.txt"

# FULL mode ceiling: 50 MB is effectively no truncation for this benchmark.
MAX_FULL_CHARS = 50 * 1024 * 1024
# TOC: max number of section entries returned.
MAX_TOC_ENTRIES = 50
# Summary: how many chars of result.out we hand to the summarizer at most
# (still fits in any modern context window).
MAX_SUMMARY_INPUT_CHARS = 400_000
# RAG defaults.
DEFAULT_RAG_K = 3
DEFAULT_RAG_CHUNK_CHARS = 2000
DEFAULT_RAG_EMBED_MODEL = "text-embedding-3-small"

# Summary mode uses a single fixed compressor across all agent models.
# Rationale: single independent variable — swap the agent, keep the summarizer
# constant. claude-haiku-4-5 is cheap, fast, and strong on structured extraction.
# Override with --summary-model for ablation.
DEFAULT_SUMMARY_MODEL = "claude-haiku-4-5"

# Hard wall-clock cap on a single PHREEQC process. Protects against pathological
# inputs (e.g. transport with huge step counts) hanging the whole question's
# 600s budget. Raise via env PHREEQC_TIMEOUT if a legitimate input needs longer.
PHREEQC_TIMEOUT = int(os.environ.get("PHREEQC_TIMEOUT", "200"))


# ── Per-subprocess runtime state ────────────────────────────────────────
# Each question runs in its own subprocess (see _run_in_process). This dict is
# populated at process start and then read by the execute_phreeqc handler so
# we don't have to thread the question / mode through every tool call.
_RUNTIME: dict[str, Any] = {
    "question": "",
    "context_mode": "toc",
    "summary_model": "",          # resolved to agent model at CLI time
    "rag_k": DEFAULT_RAG_K,
    "rag_chunk_chars": DEFAULT_RAG_CHUNK_CHARS,
    "rag_embed_model": DEFAULT_RAG_EMBED_MODEL,
    "log_path": "",               # set by _run_agent before dispatching tools
}


# ── System prompt ───────────────────────────────────────────────────────
#
# We reuse the PHREEQC-agent system prompts verbatim from disk:
#   agent/configs/system_prompt_MCQ.txt   (multiple-choice)
#   agent/configs/system_prompt_Fill.txt  (numeric / short-text fill-in)
#
# The context_mode does NOT affect the system prompt — all four modes
# (toc / full / summary / rag) share the same prompt for a given
# task_type. The modes differ only in what execute_phreeqc returns at
# runtime, not in what the agent is told about it.

SYSTEM_PROMPT_MCQ = (AGENT_CONFIGS_DIR / "system_prompt_MCQ.txt").read_text(encoding="utf-8").strip()
SYSTEM_PROMPT_FILL = (AGENT_CONFIGS_DIR / "system_prompt_Fill.txt").read_text(encoding="utf-8").strip()


def _build_system_prompt(context_mode: str, task_type: str) -> str:
    """Return the system prompt for a given task_type.

    context_mode is accepted for API compatibility but has no effect on
    the prompt — all modes share the same prompt per task_type.
    """
    return SYSTEM_PROMPT_MCQ if task_type == "mcq" else SYSTEM_PROMPT_FILL


# ── Path validation & simple file tools ─────────────────────────────────

def _validate_path(workspace: str, path: str) -> str:
    ws = Path(workspace).resolve()
    full = (ws / path).resolve()
    if ws not in (full, *full.parents):
        raise ValueError(f"Path escapes workspace: {path}")
    return str(full)


def tool_write_file(workspace: str, path: str, new_content: str) -> dict:
    try:
        full = _validate_path(workspace, path)
        Path(full).parent.mkdir(parents=True, exist_ok=True)
        Path(full).write_text(new_content, encoding="utf-8")
        return {"ok": True, "message": f"File content successfully written to: {full}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def tool_read_file(workspace: str, path: str, start_line: str = "", end_line: str = "") -> dict:
    MAX_CHARS = 20_000
    PREVIEW_LINES = 80
    try:
        full = _validate_path(workspace, path)
        if not Path(full).exists():
            return {"ok": False, "error": f"File not found: {path}"}
        content = Path(full).read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines(keepends=True)
        total = len(lines)
        sl = int(start_line) if start_line else None
        el = int(end_line) if end_line else None
        if sl and el:
            selected = lines[sl - 1:el]
            return {"ok": True, "total_lines": total,
                    "showing": f"lines {sl}-{el}", "content": "".join(selected)}
        if len(content) <= MAX_CHARS:
            return {"ok": True, "total_lines": total, "content": content}
        head = "".join(lines[:PREVIEW_LINES])
        tail = "".join(lines[-PREVIEW_LINES:])
        return {
            "ok": True, "total_lines": total,
            "showing": f"first {PREVIEW_LINES} + last {PREVIEW_LINES} lines (file too large)",
            "content": head + f"\n\n... [{total - 2 * PREVIEW_LINES} lines omitted] ...\n\n" + tail,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def tool_list_file(workspace: str, path: str = ".") -> dict:
    try:
        full = _validate_path(workspace, path)
        return {"ok": True, "entries": sorted(os.listdir(full))}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


# ── TOC builder ─────────────────────────────────────────────────────────

def _build_toc(filepath: Path) -> list[dict[str, Any]]:
    toc: list[dict[str, Any]] = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            prev_dashes_line = 0
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                is_dashes = bool(re.match(r"^-{3,}$", stripped))
                m = re.match(r"^-{3,}\s*([A-Za-z].+?)\s*-{3,}\s*$", stripped)
                if m:
                    toc.append({"line": lineno, "section": m.group(1).strip()})
                    prev_dashes_line = 0
                    continue
                if is_dashes:
                    prev_dashes_line = lineno
                    continue
                if prev_dashes_line:
                    toc.append({"line": lineno, "section": stripped.rstrip(".")})
                    prev_dashes_line = 0
                else:
                    prev_dashes_line = 0
    except Exception:  # noqa: BLE001
        pass
    if len(toc) > MAX_TOC_ENTRIES:
        half = MAX_TOC_ENTRIES // 2
        return (toc[:half]
                + [{"line": -1, "section": f"... {len(toc) - MAX_TOC_ENTRIES} sections omitted ..."}]
                + toc[-half:])
    return toc


# ── Auxiliary LLM / embedding usage logging ─────────────────────────────
#
# summary mode triggers an extra LLM call per execute_phreeqc invocation; rag
# mode triggers an extra embedding call. Those tokens are billed to the same
# run and MUST be counted in the run's total_input_tokens / total_output_tokens
# — otherwise the context-mode comparison is unfair to `full` / `toc`, which
# have no auxiliary calls. We emit a dedicated `aux_llm_usage` event to each
# question's chat log so the run aggregator can pick it up alongside the agent
# `assistant` events.

def _log_aux_usage(source: str, model: str,
                   input_tokens: int, output_tokens: int = 0) -> None:
    log_path = _RUNTIME.get("log_path") or ""
    if not log_path:
        return
    try:
        _log_event(Path(log_path), "aux_llm_usage",
                   source=source, model=model,
                   usage={"input_tokens": int(input_tokens or 0),
                          "output_tokens": int(output_tokens or 0)})
    except Exception:  # noqa: BLE001
        # Never let accounting failures kill the agent.
        pass


# ── Summary handler ─────────────────────────────────────────────────────

_SUMMARY_SYSTEM = (
    "You are a concise technical summarizer for PHREEQC geochemical simulation output. "
    "Another LLM will answer a question using ONLY your summary, so include every "
    "numeric value, species, phase, saturation index, activity, pH, equilibrium state, "
    "or computed quantity that could plausibly be relevant to the question. "
    "Omit boilerplate (headers, licensing, input echo) and non-final iteration steps. "
    "Keep your summary under 400 words. Use plain prose with inline numbers."
)


def _summarize_result(question: str, content: str, model: str) -> str:
    """Question-conditioned summary via litellm. On failure, fall back to a head/tail snippet.

    The auxiliary LLM call's input/output tokens are logged via `_log_aux_usage`
    so the run aggregator can fold them into the total token accounting.
    """
    truncated = content
    if len(truncated) > MAX_SUMMARY_INPUT_CHARS:
        half = MAX_SUMMARY_INPUT_CHARS // 2
        truncated = (truncated[:half]
                     + f"\n\n... [TRUNCATED for summarization: {len(content)} total chars] ...\n\n"
                     + truncated[-half:])
    user = f"Question the summary will be used to answer:\n{question}\n\nPHREEQC result.out:\n{truncated}"
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 1500,
    }
    # Some reasoning models reject temperature != default.
    if "claude" in model.lower() or "anthropic" in model.lower():
        kwargs["temperature"] = 0
    try:
        resp = litellm.completion(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage:
            _log_aux_usage(
                "summary", model,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("Summary model failed: %s; falling back to head/tail snippet.", e)
        head = content[:4000]
        tail = content[-4000:]
        return ("[summary model failed; head+tail snippet returned]\n\n"
                f"--- HEAD ---\n{head}\n\n--- TAIL ---\n{tail}")


# ── RAG handler ─────────────────────────────────────────────────────────

def _chunk_by_sections(content: str, toc: list[dict[str, Any]],
                       fallback_size: int) -> list[dict[str, Any]]:
    """Chunk result.out by TOC boundaries; fall back to fixed-size chunks if TOC is empty."""
    lines = content.splitlines(keepends=True)
    valid_toc = [e for e in toc if e.get("line") and e["line"] > 0]
    chunks: list[dict[str, Any]] = []
    if valid_toc:
        boundaries = [(e["line"], e["section"]) for e in valid_toc]
        for i, (start_line, section) in enumerate(boundaries):
            end_line = boundaries[i + 1][0] - 1 if i + 1 < len(boundaries) else len(lines)
            # 1-based → 0-based slice
            body = "".join(lines[max(0, start_line - 1):end_line])
            chunks.append({
                "section": section,
                "start_line": start_line,
                "end_line": end_line,
                "text": body,
            })
    if not chunks:
        # Fixed-size fallback.
        for start in range(0, len(content), fallback_size):
            chunks.append({
                "section": f"chunk_{start // fallback_size}",
                "start_line": None,
                "end_line": None,
                "text": content[start:start + fallback_size],
            })
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    import math as _m
    num = sum(x * y for x, y in zip(a, b))
    da = _m.sqrt(sum(x * x for x in a))
    db = _m.sqrt(sum(x * x for x in b))
    return num / (da * db) if da and db else 0.0


def _embed(texts: list[str], model: str) -> list[list[float]]:
    """Embed a list of texts via OpenAI embeddings API (one batched call).

    The embedding call's token usage is logged via `_log_aux_usage`.
    Embedding APIs only consume input tokens (no completion).
    """
    from openai import OpenAI
    client = OpenAI()
    # Clip each text to ~8k tokens worth; we approximate via char cap.
    safe = [t if len(t) < 24_000 else t[:24_000] for t in texts]
    resp = client.embeddings.create(model=model, input=safe)
    usage = getattr(resp, "usage", None)
    if usage:
        _log_aux_usage(
            "rag_embed", model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=0,
        )
    return [d.embedding for d in resp.data]


def _rag_select(question: str, content: str, toc: list[dict[str, Any]],
                k: int, chunk_chars: int, embed_model: str) -> list[dict[str, Any]]:
    chunks = _chunk_by_sections(content, toc, chunk_chars)
    if not chunks:
        return []
    texts = [c["text"] for c in chunks]
    try:
        embs = _embed([question] + texts, embed_model)
    except Exception as e:  # noqa: BLE001
        logger.warning("Embedding call failed: %s; returning first-k chunks.", e)
        return chunks[:k]
    q_vec = embs[0]
    scored = []
    for c, v in zip(chunks, embs[1:]):
        scored.append((c, _cosine(q_vec, v)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{**c, "similarity": round(s, 4)} for c, s in scored[:k]]


# ── execute_phreeqc dispatcher (mode-aware) ─────────────────────────────

def tool_execute_phreeqc(workspace: str, input_path: str) -> dict:
    mode: str = _RUNTIME["context_mode"]
    question: str = _RUNTIME["question"]

    try:
        full_in = _validate_path(workspace, input_path)
        if not Path(full_in).exists():
            return {"ok": False, "error": f"Input file not found: {input_path}"}

        workdir = Path(full_in).parent
        out_file = workdir / "result.out"
        files_before = set(os.listdir(workdir))

        cmd = [PHREEQC_BIN, str(full_in), str(out_file), str(DEFAULT_DB)]
        try:
            p = subprocess.run(cmd, cwd=str(workdir),
                               capture_output=True, text=True,
                               timeout=PHREEQC_TIMEOUT)
        except subprocess.TimeoutExpired as te:
            # subprocess.run already killed the child before raising.
            return {
                "ok": False,
                "returncode": -1,
                "error": (f"PHREEQC exceeded {PHREEQC_TIMEOUT}s wall-clock and was killed. "
                          "Simplify the input (fewer time steps, looser tolerances, smaller "
                          "transport grid) and try again."),
                "stdout": (te.stdout or "")[-4000:] if te.stdout else "",
                "stderr": (te.stderr or "")[-4000:] if te.stderr else "",
                "timeout_s": PHREEQC_TIMEOUT,
            }

        files_after = set(os.listdir(workdir))
        output_files = sorted(f for f in (files_after - files_before)
                              if (workdir / f).is_file())
        file_sizes = {f: (workdir / f).stat().st_size
                      for f in output_files if (workdir / f).exists()}

        result: dict[str, Any] = {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout[-4000:],
            "stderr": p.stderr[-4000:],
            "output_files": output_files,
            "file_sizes": file_sizes,
        }

        if not out_file.exists():
            return result

        content = out_file.read_text(encoding="utf-8", errors="replace")
        toc = _build_toc(out_file) if out_file.stat().st_size <= MAX_FULL_CHARS else []

        if mode == "full":
            if len(content) > MAX_FULL_CHARS:
                half = MAX_FULL_CHARS // 2
                result["result_out_content"] = (
                    content[:half]
                    + f"\n\n... [TRUNCATED: {len(content)} total chars] ...\n\n"
                    + content[-half:])
                result["warning"] = f"result.out truncated from {len(content)} to ~{MAX_FULL_CHARS} chars."
            else:
                result["result_out_content"] = content

        elif mode == "toc":
            result["result_out_toc"] = toc
            result["hint"] = "Use read_file with start_line/end_line to read specific sections."

        elif mode == "summary":
            result["result_out_summary"] = _summarize_result(
                question, content, _RUNTIME["summary_model"])
            result["hint"] = "Use read_file to inspect specific sections if the summary is insufficient."

        elif mode == "rag":
            top = _rag_select(question, content, toc,
                              k=_RUNTIME["rag_k"],
                              chunk_chars=_RUNTIME["rag_chunk_chars"],
                              embed_model=_RUNTIME["rag_embed_model"])
            result["result_out_toc"] = toc
            result["result_out_rag_hits"] = top
            result["hint"] = ("Top-k retrieved excerpts are in result_out_rag_hits, "
                              "ordered by relevance. Use read_file for additional sections.")
        else:
            return {"ok": False, "error": f"Unknown context_mode: {mode}"}

        return result

    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ── Tool specs + dispatch ───────────────────────────────────────────────

TOOL_DISPATCH = {
    "write_file": lambda ws, args: tool_write_file(ws, **args),
    "read_file": lambda ws, args: tool_read_file(ws, **args),
    "list_file": lambda ws, args: tool_list_file(ws, **args),
    "execute_phreeqc": lambda ws, args: tool_execute_phreeqc(ws, **args),
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a text file in the workspace.",
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string", "description": "Relative path to the file."},
                           "new_content": {"type": "string", "description": "Content to write."},
                       },
                       "required": ["path", "new_content"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file. Optionally specify start_line/end_line (1-based) for a range.",
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string"},
                           "start_line": {"type": "string"},
                           "end_line": {"type": "string"},
                       },
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_file",
        "description": "List files and directories in the workspace.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "execute_phreeqc",
        "description": ("Run PHREEQC on an input file. Returns execution status, "
                        "stdout/stderr, output file paths, and result content."),
        "parameters": {"type": "object",
                       "properties": {"input_path": {"type": "string"}},
                       "required": ["input_path"]}}},
]


# ── Logging ─────────────────────────────────────────────────────────────

def _log_event(log_path: Path, action: str, **data: Any) -> None:
    event = {"ts": datetime.now().isoformat(timespec="seconds"), "action": action, **data}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ── Agent loop ──────────────────────────────────────────────────────────

def _is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower() or "anthropic" in model.lower()


def _run_agent(prompt: str, workspace: str, model: str, max_steps: int,
               system_prompt: str) -> None:
    log_dir = Path(workspace) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"chat_{ts}_{int(time.time())}.jsonl"

    # Expose to aux helpers (summary / rag_embed) so they can log their
    # token usage into the same per-question chat log.
    _RUNTIME["log_path"] = str(log_path)

    _log_event(log_path, "run_start", model=model, max_steps=max_steps,
               context_mode=_RUNTIME["context_mode"])
    _log_event(log_path, "user", content=prompt)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    anthropic = _is_anthropic_model(model)

    for step in range(1, max_steps + 1):
        if anthropic and step == max_steps - 1:
            messages.append({
                "role": "user",
                "content": ("WARNING: You have only 2 steps remaining. "
                            "You MUST write final_answer.txt NOW with your best answer. "
                            "Use write_file to create final_answer.txt."),
            })

        # Per-step API call with retries.
        for attempt in range(1, 11):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "tools": TOOL_SPECS,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
                if anthropic:
                    kwargs["temperature"] = 0
                    kwargs["max_tokens"] = 4096
                response = litellm.completion(**kwargs)
                break
            except Exception as api_err:  # noqa: BLE001
                err = str(api_err).lower()
                transient = any(k in err for k in
                                ("429", "500", "502", "503", "529", "timeout",
                                 "connection", "overloaded", "rate_limit"))
                if not transient or attempt == 10:
                    _log_event(log_path, "run_end", status="api_error", step=step, error=str(api_err))
                    raise
                wait = min(15 * attempt, 90)
                logger.warning("API call failed (attempt %d/10, wait %ds): %s", attempt, wait, api_err)
                time.sleep(wait)

        time.sleep(1)
        choice = response.choices[0]
        assistant_msg = choice.message
        tool_calls = assistant_msg.tool_calls or []

        usage_dict: dict[str, int] = {}
        if response.usage:
            usage_dict = {
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
            }

        first_tc = tool_calls[0] if tool_calls else None
        _log_event(
            log_path, "assistant", step=step,
            content=assistant_msg.content or "",
            usage=usage_dict,
            tool_calls=[{"id": first_tc.id, "name": first_tc.function.name,
                         "arguments": first_tc.function.arguments}] if first_tc else [],
        )

        payload: dict[str, Any] = {"role": "assistant", "content": assistant_msg.content or ""}
        if first_tc:
            payload["tool_calls"] = [{
                "id": first_tc.id,
                "type": "function",
                "function": {"name": first_tc.function.name,
                             "arguments": first_tc.function.arguments or "{}"},
            }]
        messages.append(payload)

        if first_tc is None:
            _log_event(log_path, "run_end", status="answered", step=step)
            return

        tool_name = first_tc.function.name
        try:
            tool_args = json.loads(first_tc.function.arguments or "{}")
        except json.JSONDecodeError:
            tool_args = {}

        dispatch = TOOL_DISPATCH.get(tool_name)
        if dispatch is None:
            tool_result: dict[str, Any] = {"ok": False, "error": f"Unknown tool: {tool_name}"}
        else:
            try:
                tool_result = dispatch(workspace, tool_args)
            except Exception as e:  # noqa: BLE001
                tool_result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        messages.append({
            "role": "tool",
            "tool_call_id": first_tc.id,
            "content": json.dumps(tool_result, ensure_ascii=False),
        })
        _log_event(log_path, "tool", step=step, tool=tool_name, args=tool_args, result=tool_result)

    _log_event(log_path, "run_end", status="max_steps_reached", step=max_steps)


# ── Per-question subprocess wrapper ─────────────────────────────────────

def _run_in_process(row: dict[str, Any], result_queue: multiprocessing.Queue,
                    ws_root: str, model: str, max_steps: int,
                    runtime_overrides: dict[str, Any]) -> None:
    try:
        # Populate runtime state for this subprocess.
        _RUNTIME.update(runtime_overrides)
        _RUNTIME["question"] = row["question"]

        idx = row["index"]
        workspace = os.path.join(ws_root, f"dataset_q_{idx + 1:05d}")
        os.makedirs(workspace, exist_ok=True)

        if DB_REFERENCE.exists():
            shutil.copy(DB_REFERENCE, os.path.join(workspace, "database_reference.txt"))

        system_prompt = _build_system_prompt(_RUNTIME["context_mode"], row["task_type"])
        prompt = build_prompt(row, for_agent=True)
        _run_agent(prompt, workspace, model, max_steps, system_prompt)
    except Exception as e:  # noqa: BLE001
        result_queue.put({"error": str(e)})
        return
    result_queue.put({"ok": True})


def _process_one(row: dict[str, Any], ws_root: str, model: str,
                 max_steps: int, runtime_overrides: dict[str, Any],
                 timeout_s: int) -> dict[str, Any]:
    idx = row["index"]
    q: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_in_process,
        args=(row, q, ws_root, model, max_steps, runtime_overrides),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.kill()
        proc.join()
        result = grade_row(row, pred=None)
        result["error"] = f"Timeout {timeout_s}s"
        return result

    if proc.exitcode != 0:
        err = "Process crashed"
        if not q.empty():
            m = q.get_nowait()
            if isinstance(m, dict) and m.get("error"):
                err = m["error"]
        result = grade_row(row, pred=None)
        result["error"] = err
        return result

    ws = Path(ws_root) / f"dataset_q_{idx + 1:05d}"
    fa = ws / "final_answer.txt"
    pred = parse_answer(fa.read_text(encoding="utf-8"), row["task_type"], strict=True) if fa.exists() else None
    return grade_row(row, pred)


# ── CLI ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Custom agent with switchable context mode.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--model", required=True,
                   help="litellm model string, e.g. claude-opus-4-6, gpt-5.2, gemini/gemini-2.5-pro")
    p.add_argument("--name", required=True, help="Run name for result subfolder.")
    p.add_argument("--context-mode", choices=("full", "toc", "summary", "rag"),
                   required=True, help="Tool-return handler for execute_phreeqc.")
    p.add_argument("--max-steps", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-question wall-clock timeout (seconds).")
    p.add_argument("--resume", action="store_true",
                   help="Skip questions with an existing final_answer.txt.")
    p.add_argument("--task-filter", choices=("all", "mcq", "fill"), default="all")
    # Summary-mode knobs
    p.add_argument("--summary-model", default=None,
                   help=f"litellm model used for summary mode (default: "
                        f"{DEFAULT_SUMMARY_MODEL}, held constant across all "
                        f"agent models). Override for ablation.")
    # RAG-mode knobs
    p.add_argument("--rag-k", type=int, default=DEFAULT_RAG_K,
                   help="Top-k chunks returned in RAG mode (default: %(default)s).")
    p.add_argument("--rag-chunk-chars", type=int, default=DEFAULT_RAG_CHUNK_CHARS,
                   help="Fallback chunk size (chars) when no TOC is available (default: %(default)s).")
    p.add_argument("--rag-embed-model", default=DEFAULT_RAG_EMBED_MODEL,
                   help="OpenAI embedding model (default: %(default)s).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    records = load_dataset(dataset_path)
    if args.task_filter != "all":
        records = [r for r in records if r["task_type"] == args.task_filter]
        if not records:
            raise RuntimeError(f"No rows with task_type={args.task_filter} in {dataset_path}")

    run_dir = RESULT_ROOT / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ws_root = run_dir / "work_space"
    ws_root.mkdir(parents=True, exist_ok=True)

    # Summary uses a single fixed compressor across all agent models.
    resolved_summary_model = args.summary_model or DEFAULT_SUMMARY_MODEL

    runtime_overrides = {
        "context_mode": args.context_mode,
        "summary_model": resolved_summary_model,
        "rag_k": args.rag_k,
        "rag_chunk_chars": args.rag_chunk_chars,
        "rag_embed_model": args.rag_embed_model,
    }

    results: list[dict[str, Any]] = []
    records_to_run = records

    if args.resume:
        remaining: list[dict[str, Any]] = []
        for row in records:
            idx = row["index"]
            fa = ws_root / f"dataset_q_{idx + 1:05d}" / "final_answer.txt"
            if fa.exists():
                pred = parse_answer(fa.read_text(encoding="utf-8"), row["task_type"], strict=True)
                results.append(grade_row(row, pred))
            else:
                remaining.append(row)
        print(f"Resuming: {len(results)} done, {len(remaining)} remaining")
        records_to_run = remaining

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(_process_one, row, str(ws_root), args.model,
                      args.max_steps, runtime_overrides, args.timeout): row
            for row in records_to_run
        }
        it = (tqdm(as_completed(futures), total=len(futures),
                   desc=f"custom/{args.context_mode} ({args.model})", unit="q")
              if tqdm else as_completed(futures))
        for fut in it:
            results.append(fut.result())

    results.sort(key=lambda r: r["index"])
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )

    agg = summarize_results(results)

    # Token aggregation from per-question chat logs.
    # Includes BOTH the main agent's `assistant` events AND any auxiliary LLM /
    # embedding calls (`aux_llm_usage` events) triggered by summary / rag modes.
    # This keeps the comparison between context modes fair — summary / rag don't
    # get a free pass for their extra LLM calls.
    total_input = total_output = 0
    agent_input = agent_output = 0
    aux_by_source: dict[str, dict[str, int]] = {}
    for qdir in ws_root.glob("dataset_q_*"):
        for logf in qdir.glob("logs/chat_*.jsonl"):
            with logf.open() as lf:
                for line in lf:
                    try:
                        e = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    action = e.get("action")
                    usage = e.get("usage")
                    if not isinstance(usage, dict):
                        continue
                    in_t = usage.get("input_tokens", 0) or 0
                    out_t = usage.get("output_tokens", 0) or 0
                    if action == "assistant":
                        agent_input += in_t
                        agent_output += out_t
                        total_input += in_t
                        total_output += out_t
                    elif action == "aux_llm_usage":
                        src = e.get("source", "aux")
                        slot = aux_by_source.setdefault(src, {"input_tokens": 0,
                                                              "output_tokens": 0,
                                                              "calls": 0,
                                                              "model": e.get("model", "")})
                        slot["input_tokens"] += in_t
                        slot["output_tokens"] += out_t
                        slot["calls"] += 1
                        # Keep model name (should be constant per source within a run).
                        if not slot.get("model") and e.get("model"):
                            slot["model"] = e["model"]
                        total_input += in_t
                        total_output += out_t

    summary: dict[str, Any] = {
        "model": args.model,
        "method": f"custom_agent_{args.context_mode}",
        "context_mode": args.context_mode,
        "dataset": dataset_path.name,
        "run_name": args.name,
        "max_steps": args.max_steps,
        "workers": args.workers,
        "task_filter": args.task_filter,
        **agg,
        # ── token accounting ──────────────────────────────────────────────
        # total_* includes BOTH agent and aux (summary / rag_embed) calls.
        # agent_* is the main agent only.
        # aux_by_source breaks out every auxiliary model so readers can see
        # the compressor / embedder contributions separately.
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "agent_input_tokens": agent_input,
        "agent_output_tokens": agent_output,
        "aux_by_source": aux_by_source,   # {source: {model, input_tokens, output_tokens, calls}}
    }

    # ── mode-specific metadata (only include fields relevant to this mode) ──
    if args.context_mode == "summary":
        summary["summary_model"] = runtime_overrides["summary_model"]
    elif args.context_mode == "rag":
        summary["rag_embed_model"] = args.rag_embed_model
        summary["rag_k"] = args.rag_k
        summary["rag_chunk_chars"] = args.rag_chunk_chars
    elif args.context_mode == "full":
        summary["max_full_chars"] = MAX_FULL_CHARS
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
