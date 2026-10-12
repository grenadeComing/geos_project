"""
Custom agent evaluation runner. This is the primary script for comparing
context-management strategies at the tool-return boundary.

Supported context modes (via --context-mode):

  full     Raw result.out returned inline, truncated at MAX_FULL_CHARS.
           The default 100k-char cap matches the ACM rebuttal raw baseline.
  toc      Section-level table of contents returned; agent uses read_file
           with start_line/end_line to inspect sections. (Our TOC pattern.)
  summary  Question-conditioned LLM summary of result.out returned.
  rag      Embedding-based retrieval of top-k chunks from result.out,
           ranked by cosine similarity to the question.
  grep     Metadata only returned (status, file paths/sizes, line count);
           the agent locates evidence with a grep_file regex-search tool
           and reads line ranges via read_file. (Keyword-search baseline;
           grep_file is exposed to the model only in this mode.)
  toc_report  Uncapped TOC (as in `toc` with a large cap) plus an instruction to
           answer from result.out as printed rather than from SELECTED_OUTPUT files.
           Tests whether reminding the agent to read the report removes the TOC loss.
  toc_round   Uncapped TOC plus an instruction to round values with more digits than the
           options to the nearest option-precision value (no truncation). Names no file:
           separates the rounding warning from naming the authoritative source.
  hybrid   Raw result.out inline when it fits MAX_FULL_CHARS (as in `full`);
           otherwise an uncapped section TOC (as in `toc`), so the agent
           retrieves only the long outputs.

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

# All NeurIPS-paper runs use the clean 200-item benchmark (matches the
# "dataset" field recorded in every result/custom/*/summary.json).
DEFAULT_DATASET = REPO_ROOT / "datasets" / "neurips_dataset_200_clean.jsonl"
RESULT_ROOT = REPO_ROOT / "result" / "custom"

# Agent configs (reuse the original PHREEQC-agent system prompt file).
AGENT_CONFIGS_DIR = REPO_ROOT / "agent" / "configs"

# PHREEQC executable + database. The benchmark uses the updated phreeqc.dat
# supplied by the dataset author, not the stock PHREEQC installation database.
PHREEQC_BIN = os.environ.get(
    "PHREEQC_BIN",
    "/Users/kezhang/Desktop/projects/geos/phreeqc-3.8.6-17100/src/phreeqc",
)
DEFAULT_DB = (
    Path(os.environ["PHREEQC_DB"]).expanduser()
    if os.environ.get("PHREEQC_DB")
    else REPO_ROOT / "archive_acm_2026" / "all_materials" / "phreeqc.dat"
)
if not DEFAULT_DB.exists():
    raise FileNotFoundError(f"PHREEQC database not found: {DEFAULT_DB}. Set PHREEQC_DB to a valid database path.")
DB_REFERENCE = REPO_ROOT / "archive_acm_2026" / "all_materials" / "database_reference.txt"

# FULL/raw mode ceiling. Default matches the ACM rebuttal raw baseline.
# Override with --max-full-chars or MAX_FULL_CHARS=... for cap ablations.
DEFAULT_MAX_FULL_CHARS = int(os.environ.get("MAX_FULL_CHARS", "100000"))
# TOC: max number of section entries returned.
MAX_TOC_ENTRIES = 50
# Summary: how many chars of result.out we hand to the summarizer at most
# (still fits in any modern context window).
MAX_SUMMARY_INPUT_CHARS = 400_000
# RAG defaults.
DEFAULT_RAG_K = 3
DEFAULT_RAG_CHUNK_CHARS = 2000
DEFAULT_RAG_EMBED_MODEL = "text-embedding-3-small"
# Grep defaults. Return-size cap mirrors read_file's MAX_CHARS so neither
# navigation tool can smuggle an unbounded payload into context.
DEFAULT_GREP_MAX_MATCHES = 20
DEFAULT_GREP_CONTEXT_LINES = 2
GREP_MAX_RETURN_CHARS = 20_000

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
    "max_full_chars": DEFAULT_MAX_FULL_CHARS,
    "log_path": "",               # set by _run_agent before dispatching tools
}


# ── System prompt ───────────────────────────────────────────────────────
#
# We reuse the PHREEQC-agent system prompts from disk, but rewrite the
# execute_phreeqc contract to match the context mode under evaluation:
#   agent/configs/system_prompt_MCQ.txt   (multiple-choice)
#   agent/configs/system_prompt_Fill.txt  (numeric fill-in)
#
# This avoids giving full / summary / rag runs stale TOC-specific instructions.

SYSTEM_PROMPT_MCQ = (AGENT_CONFIGS_DIR / "system_prompt_MCQ.txt").read_text(encoding="utf-8").strip()
SYSTEM_PROMPT_FILL = (AGENT_CONFIGS_DIR / "system_prompt_Fill.txt").read_text(encoding="utf-8").strip()

_EXECUTE_TOOL_LINE = (
    "- execute_phreeqc(input_path): Run PHREEQC on an input file. Returns "
    "metadata and a TOC for result.out. File contents are NOT included — "
    "use read_file to inspect results."
)

_EXECUTE_TOOL_CONTRACTS = {
    "full": (
        "- execute_phreeqc(input_path): Run PHREEQC on an input file. Returns "
        "metadata and raw result.out content inline, truncated if very large. "
        "Use read_file only if you need to inspect an additional output file "
        "or re-read a specific portion."
    ),
    "toc": _EXECUTE_TOOL_LINE,
    "toc_report": (
        _EXECUTE_TOOL_LINE + "\n"
        "IMPORTANT: The authoritative values are in result.out, printed at the precision the answer options use. Before answering, read the section of result.out that contains the quantity you need and take the value exactly as printed there. Do not answer from SELECTED_OUTPUT or USER_PUNCH files: they print unrounded values that can match a neighboring option."
    ),
    "toc_round": (
        _EXECUTE_TOOL_LINE + "\n"
        "IMPORTANT: The answer options give each value at a fixed precision. If a value you read has more digits than the options, round it to the options' precision to the nearest value; do not truncate digits. Choose the option that equals the correctly rounded value."
    ),
    "hybrid": (
        "- execute_phreeqc(input_path): Run PHREEQC on an input file. Returns "
        "metadata plus the raw result.out content inline when it is small enough. "
        "For a large result.out it returns a TOC instead (section names with "
        "starting line numbers) and no content; in that case use read_file with "
        "start_line/end_line to read the sections you need."
    ),
    "summary": (
        "- execute_phreeqc(input_path): Run PHREEQC on an input file. Returns "
        "metadata and a question-conditioned summary of result.out. File "
        "contents are not included by default; use read_file to inspect "
        "specific output sections if the summary is insufficient."
    ),
    "rag": (
        "- execute_phreeqc(input_path): Run PHREEQC on an input file. Returns "
        "metadata, a TOC for result.out, and top-k retrieved excerpts selected "
        "for the current question. Use read_file with start_line/end_line for "
        "additional sections if the retrieved excerpts are insufficient."
    ),
    # grep mode replaces the single execute line with two lines: the metadata-only
    # execute contract plus the grep_file tool doc (grep_file exists only in this
    # mode, so the base system prompt files never mention it).
    "grep": (
        "- execute_phreeqc(input_path): Run PHREEQC on an input file. Returns "
        "metadata only (status, output file paths and sizes, result.out line "
        "count). File contents are NOT included — locate evidence with grep_file, "
        "then read specific line ranges with read_file.\n"
        "- grep_file(path, pattern, max_matches?, context_lines?, case_sensitive?): "
        "Search a text file line-by-line with a regular expression "
        "(case-insensitive by default). Returns matching line numbers with a few "
        "lines of surrounding context. Use it to find sections, species, or "
        "numeric fields in result.out, then read_file with start_line/end_line "
        "to inspect the full region.\n"
        "NOTE: In this mode, read_file on a large file WITHOUT start_line/end_line "
        "returns no content. Always grep_file first to find line numbers, then "
        "read the range you need."
    ),
}


def _build_system_prompt(context_mode: str, task_type: str) -> str:
    """Return the system prompt with a mode-accurate execute_phreeqc contract."""
    base = SYSTEM_PROMPT_MCQ if task_type == "mcq" else SYSTEM_PROMPT_FILL
    contract = _EXECUTE_TOOL_CONTRACTS[context_mode]
    if _EXECUTE_TOOL_LINE in base:
        base = base.replace(_EXECUTE_TOOL_LINE, contract)
    else:
        base = base + "\n" + contract
    return (
        base
        + "\n\n# Context mode for this run\n"
        + f"The execute_phreeqc tool is configured in `{context_mode}` mode. "
        + "Use the payload it returns according to the tool contract above."
    )


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
        grep_mode = _RUNTIME.get("context_mode") == "grep"
        if grep_mode and (sl or el) and not (sl and el):
            # Honor half-open ranges instead of silently falling through to a
            # whole-file read (which the grep protocol gate below would refuse).
            if sl and not el:
                el = min(total, sl + 199)
            else:
                sl = max(1, el - 199)
        if sl and el:
            selected = lines[sl - 1:el]
            return {"ok": True, "total_lines": total,
                    "showing": f"lines {sl}-{el}", "content": "".join(selected)}
        if len(content) <= MAX_CHARS:
            return {"ok": True, "total_lines": total, "content": content}
        if grep_mode:
            # Grep protocol: no free head/tail preview on oversized files —
            # otherwise unranged read_file becomes an inline-payload escape
            # hatch and the run stops measuring keyword-search navigation.
            # (Small files, e.g. SELECTED_OUTPUT .sel files, read normally.)
            return {"ok": False, "total_lines": total,
                    "error": (f"File is too large ({total} lines) to read without a line "
                              "range in grep mode. Use grep_file(path, pattern) to locate "
                              "the relevant lines, then call read_file with "
                              "start_line/end_line.")}
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


def tool_grep_file(workspace: str, path: str, pattern: str,
                   max_matches: Any = None, context_lines: Any = None,
                   case_sensitive: Any = None) -> dict:
    """Regex line-search over a workspace file (the `grep` context mode's
    navigation primitive). Returns matching line numbers with ±context_lines
    of surrounding text. When more than max_matches lines match, the first
    and last halves are kept (same convention as the TOC builder's section
    trimming) since PHREEQC final results concentrate at the file tail."""
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    try:
        full = _validate_path(workspace, path)
        if not Path(full).exists():
            return {"ok": False, "error": f"File not found: {path}"}

        mm = max(1, _coerce_int(max_matches, DEFAULT_GREP_MAX_MATCHES))
        cl = max(0, _coerce_int(context_lines, DEFAULT_GREP_CONTEXT_LINES))
        cs = str(case_sensitive).strip().lower() in ("true", "1", "yes")
        try:
            rx = re.compile(pattern, 0 if cs else re.IGNORECASE)
        except re.error as e:
            return {"ok": False, "error": f"Invalid regular expression: {e}"}

        lines = Path(full).read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)
        hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
        total_matches = len(hits)
        if not total_matches:
            return {"ok": True, "total_lines": total_lines, "total_matches": 0,
                    "matches": [],
                    "hint": "No lines matched. Try a shorter pattern or different keyword."}

        clipped = total_matches > mm
        if clipped:
            half = max(1, mm // 2)
            hits = hits[:half] + hits[-(mm - half):]

        matches: list[dict[str, Any]] = []
        budget = GREP_MAX_RETURN_CHARS
        for i in hits:
            lo, hi = max(0, i - cl), min(total_lines, i + cl + 1)
            block = "\n".join(f"{n + 1}: {lines[n]}" for n in range(lo, hi))
            if budget - len(block) < 0:
                clipped = True
                break
            budget -= len(block)
            matches.append({"line": i + 1, "context": block})

        result: dict[str, Any] = {"ok": True, "total_lines": total_lines,
                                  "total_matches": total_matches,
                                  "showing": len(matches), "matches": matches}
        if clipped:
            result["truncated"] = True
            result["hint"] = (f"{total_matches} lines matched; showing {len(matches)} "
                              "(earliest and latest). Narrow the pattern, or use "
                              "read_file with start_line/end_line around a match.")
        return result
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


def _tail_text(value: str | bytes | None, limit: int = 4000) -> str:
    """Return a JSON-serializable text tail from subprocess output."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


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
    # text-embedding-3-small's per-input limit is 8192 tokens. On dense
    # numeric PHREEQC output, 1 token ≈ 1.3 chars, so 10_000 chars stays
    # comfortably under 8192 tokens. (Earlier 24_000 cap was too loose
    # and triggered "maximum input length is 8192 tokens" 400s.)
    _CHAR_CAP = 10_000
    safe = [t if len(t) < _CHAR_CAP else t[:_CHAR_CAP] for t in texts]
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
        if out_file.exists():
            out_file.unlink()
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
                "stdout": _tail_text(te.stdout),
                "stderr": _tail_text(te.stderr),
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

        max_full_chars = int(_RUNTIME.get("max_full_chars", DEFAULT_MAX_FULL_CHARS))
        content = out_file.read_text(encoding="utf-8", errors="replace")
        toc = _build_toc(out_file) if out_file.stat().st_size <= max_full_chars else []

        if mode == "full":
            if len(content) > max_full_chars:
                half = max_full_chars // 2
                result["result_out_content"] = (
                    content[:half]
                    + f"\n\n... [TRUNCATED: {len(content)} total chars, cap={max_full_chars}] ...\n\n"
                    + content[-half:])
                result["warning"] = f"result.out truncated from {len(content)} to ~{max_full_chars} chars."
            else:
                result["result_out_content"] = content

        elif mode == "toc":
            result["result_out_toc"] = toc
            result["hint"] = "Use read_file with start_line/end_line to read specific sections."

        elif mode == "toc_report":
            # Same evidence as TOC, never capped, plus the report-reading reminder.
            result["result_out_toc"] = _build_toc(out_file)
            result["hint"] = ("Use read_file with start_line/end_line to read the section of result.out "
                              "that contains the quantity you need. The authoritative values are in result.out, printed at the precision the answer options use. Before answering, read the section of result.out that contains the quantity you need and take the value exactly as printed there. Do not answer from SELECTED_OUTPUT or USER_PUNCH files: they print unrounded values that can match a neighboring option.")

        elif mode == "toc_round":
            # Same evidence as TOC, never capped, plus a rounding reminder that names no file.
            result["result_out_toc"] = _build_toc(out_file)
            result["hint"] = ("Use read_file with start_line/end_line to read specific sections. "
                              "IMPORTANT: The answer options give each value at a fixed precision. If a value you read has more digits than the options, round it to the options' precision to the nearest value; do not truncate digits. Choose the option that equals the correctly rounded value.")

        elif mode == "hybrid":
            # Inline outputs that fit the budget; retrieve only long ones. The TOC
            # here is built without the size cap, matching the uncapped-TOC runs.
            if len(content) <= max_full_chars:
                result["result_out_content"] = content
            else:
                result["result_out_toc"] = _build_toc(out_file)
                result["result_out_total_chars"] = len(content)
                result["hint"] = ("result.out is too large to inline. Use read_file with "
                                  "start_line/end_line to read specific sections.")

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
        elif mode == "grep":
            # Keyword-search protocol: metadata only — no TOC, no content.
            # Line count is the grep-world analogue of what TOC's line numbers
            # reveal about file length (cf. `wc -l`), so the agent can target
            # the tail with read_file without a wasted probe step.
            result["result_out_total_lines"] = len(content.splitlines())
            result["hint"] = ("File contents are NOT included. Use grep_file(path, pattern) "
                              "to locate relevant lines in result.out, then read_file with "
                              "start_line/end_line to inspect the surrounding region. "
                              "read_file without a line range returns no content for large files.")
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
    "grep_file": lambda ws, args: tool_grep_file(ws, **args),
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
                        "stdout/stderr, output file paths, and the result payload "
                        "configured for this run's context mode."),
        "parameters": {"type": "object",
                       "properties": {"input_path": {"type": "string"}},
                       "required": ["input_path"]}}},
]

# grep_file is appended to the spec list ONLY in grep mode, so runs in every
# other context mode present the exact tool surface of the paper's original
# protocols (a TOC rerun with grep_file available would be a different protocol).
GREP_TOOL_SPEC = {
    "type": "function", "function": {
        "name": "grep_file",
        "description": ("Search a text file line-by-line with a regular expression "
                        "(case-insensitive by default). Returns matching line numbers "
                        "with surrounding context lines. Use read_file with "
                        "start_line/end_line to inspect a region around a match."),
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string",
                                    "description": "Relative path to the file, e.g. result.out."},
                           "pattern": {"type": "string",
                                       "description": "Regular expression to search for."},
                           "max_matches": {"type": "integer",
                                           "description": "Max matching lines to return (default 20). "
                                                          "When exceeded, earliest and latest matches are kept."},
                           "context_lines": {"type": "integer",
                                             "description": "Lines of context around each match (default 2)."},
                           "case_sensitive": {"type": "boolean",
                                              "description": "Match case-sensitively (default false)."},
                       },
                       "required": ["path", "pattern"]}}}


def _tool_specs_for_mode(mode: str) -> list[dict[str, Any]]:
    if mode == "grep":
        return TOOL_SPECS + [GREP_TOOL_SPEC]
    return TOOL_SPECS


# ── Logging ─────────────────────────────────────────────────────────────

def _log_event(log_path: Path, action: str, **data: Any) -> None:
    event = {"ts": datetime.now().isoformat(timespec="seconds"), "action": action, **data}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ── Agent loop ──────────────────────────────────────────────────────────

def _is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower() or "anthropic" in model.lower()


def _is_gemini_model(model: str) -> bool:
    m = model.lower()
    return "gemini" in m or m.startswith("google/")


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
    gemini = _is_gemini_model(model)
    tool_specs = _tool_specs_for_mode(_RUNTIME["context_mode"])

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
                    "tools": tool_specs,
                    "tool_choice": "auto",
                }
                if gemini:
                    # Gemini rejects parallel_tool_calls=False when multiple
                    # tools are available. If it returns multiple tool calls,
                    # we preserve every id and return one result per id below.
                    kwargs["drop_params"] = True
                else:
                    kwargs["parallel_tool_calls"] = False
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

        formatted_tool_calls = []
        for i, tc in enumerate(tool_calls):
            call_id = tc.id or f"call_{step}_{i}"
            formatted_tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": tc.function.name,
                             "arguments": tc.function.arguments or "{}"},
            })
        _log_event(
            log_path, "assistant", step=step,
            content=assistant_msg.content or "",
            usage=usage_dict,
            tool_calls=[{
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            } for tc in formatted_tool_calls],
        )

        payload: dict[str, Any] = {"role": "assistant", "content": assistant_msg.content or ""}
        if formatted_tool_calls:
            payload["tool_calls"] = formatted_tool_calls
        messages.append(payload)

        if not formatted_tool_calls:
            _log_event(log_path, "run_end", status="answered", step=step)
            return

        # Gemini 3 always returns a unique function-call id. Because Gemini
        # rejects parallel_tool_calls=False, preserve every returned id and
        # provide a matching tool result for each call in the same order.
        for tc in formatted_tool_calls:
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"] or "{}")
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
                "tool_call_id": tc["id"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            })
            _log_event(log_path, "tool", step=step, tool=tool_name,
                       tool_call_id=tc["id"], args=tool_args, result=tool_result)

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

    child_msg = None
    if not q.empty():
        child_msg = q.get_nowait()

    if proc.exitcode != 0:
        err = "Process crashed"
        if isinstance(child_msg, dict) and child_msg.get("error"):
            err = child_msg["error"]
        result = grade_row(row, pred=None)
        result["error"] = err
        return result

    if isinstance(child_msg, dict) and child_msg.get("error"):
        result = grade_row(row, pred=None)
        result["error"] = child_msg["error"]
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
    p.add_argument("--context-mode", choices=("full", "toc", "summary", "rag", "grep", "hybrid", "toc_report", "toc_round"),
                   required=True, help="Tool-return handler for execute_phreeqc.")
    p.add_argument("--max-steps", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-question wall-clock timeout (seconds).")
    p.add_argument("--resume", action="store_true",
                   help="Skip questions with an existing final_answer.txt.")
    p.add_argument("--task-filter", choices=("all", "mcq", "fill"), default="all")
    p.add_argument("--start-index", type=int, default=None,
                   help="Optional 1-based public dataset id to start from, inclusive.")
    p.add_argument("--end-index", type=int, default=None,
                   help="Optional 1-based public dataset id to end at, inclusive.")
    p.add_argument("--max-full-chars", type=int, default=DEFAULT_MAX_FULL_CHARS,
                   help="Raw/full mode character cap for result.out observations (default: %(default)s).")
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
    started_at = datetime.now().astimezone()
    started_perf = time.perf_counter()
    dataset_path = Path(args.dataset).expanduser().resolve()
    records = load_dataset(dataset_path)
    if args.task_filter != "all":
        records = [r for r in records if r["task_type"] == args.task_filter]
        if not records:
            raise RuntimeError(f"No rows with task_type={args.task_filter} in {dataset_path}")
    if args.start_index is not None:
        records = [r for r in records if int(r["id"]) >= args.start_index]
    if args.end_index is not None:
        records = [r for r in records if int(r["id"]) <= args.end_index]
    if not records:
        raise RuntimeError(
            f"No rows left after filters: task_filter={args.task_filter}, "
            f"start_index={args.start_index}, end_index={args.end_index}"
        )

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
        "max_full_chars": args.max_full_chars,
    }

    results: list[dict[str, Any]] = []
    records_to_run = records
    selected_ids = {int(r["id"]) for r in records}

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

    # If a range filter is used with --resume, preserve existing result rows
    # outside the selected range so patch-runs can safely replace only a suffix.
    result_path = run_dir / "results.jsonl"
    if args.resume and (args.start_index is not None or args.end_index is not None) and result_path.exists():
        preserved: list[dict[str, Any]] = []
        with result_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                old = json.loads(line)
                if int(old.get("index", -1)) not in selected_ids:
                    preserved.append(old)
        if preserved:
            print(f"Preserving {len(preserved)} existing result rows outside selected range")
            results.extend(preserved)

    results.sort(key=lambda r: r["index"])
    result_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )

    agg = summarize_results(results)
    ended_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - started_perf

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
        "start_index": args.start_index,
        "end_index": args.end_index,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "seconds_per_row": round(elapsed_seconds / len(records), 3) if records else None,
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
    elif args.context_mode in ("full", "hybrid"):
        summary["max_full_chars"] = args.max_full_chars
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
