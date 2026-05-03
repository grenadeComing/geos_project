"""
TOC Ablation: Agent with raw PHREEQC output (no TOC) or TOC mode via litellm.
Uses litellm for unified OpenAI/Anthropic tool-calling protocol.

Usage:
  # Claude Opus 4.6 — raw output (no TOC)
  python evaluate_ablation_raw.py --model claude-opus-4-6 --name sj_ablation_raw_opus46 --workers 4

  # GPT-5.2 — raw output (no TOC)
  python evaluate_ablation_raw.py --model gpt-5.2 --name sj_ablation_raw_gpt52 --workers 4

  # GPT-5.2 — TOC mode (with token logging)
  python evaluate_ablation_raw.py --model gpt-5.2 --name sj_toc_gpt52 --workers 4 --toc

Requires: pip install litellm
Requires: ANTHROPIC_API_KEY / OPENAI_API_KEY env vars
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import litellm

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "dataset_S+J.jsonl"
RESULT_ROOT = BASE_DIR / "result"
AGENT_DIR = BASE_DIR / "agent"

PHREEQC_BIN = os.environ.get("PHREEQC_BIN", "phreeqc")
DEFAULT_DB = BASE_DIR / "phreeqc.dat"

SYSTEM_PROMPT_ORIGINAL = (AGENT_DIR / "configs" / "system_prompt.txt").read_text(encoding="utf-8").strip()

# Modify system prompt: remove TOC references, tell agent output is returned directly
SYSTEM_PROMPT = SYSTEM_PROMPT_ORIGINAL.replace(
    "execute_phreeqc(input_path): Run PHREEQC on an input file. Returns metadata and a TOC for result.out. File contents are NOT included — use read_file to inspect results.",
    "execute_phreeqc(input_path): Run PHREEQC on an input file. Returns the full content of result.out directly. You may also use read_file for targeted inspection."
)

# Maximum chars of result.out to return inline.
# For the S+J dataset max result.out is ~846k chars, so 50MB is effectively
# "no truncation" for this benchmark — raw mode gets the full content.
# Matches MAX_RAW_CHARS in evaluate_claude_sdk.py so the TOC-vs-Raw ablation
# is apples-to-apples across our custom agent and the Claude Agent SDK.
MAX_RAW_CHARS = 50 * 1024 * 1024


# ── Tool implementations ──

def _validate_path(workspace: str, path: str) -> str:
    """Resolve path relative to workspace, reject escapes."""
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
    except Exception as e:
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
            text = "".join(selected)
            return {"ok": True, "total_lines": total, "showing": f"lines {sl}-{el}", "content": text}

        if len(content) <= MAX_CHARS:
            return {"ok": True, "total_lines": total, "content": content}

        head = "".join(lines[:PREVIEW_LINES])
        tail = "".join(lines[-PREVIEW_LINES:])
        return {
            "ok": True,
            "total_lines": total,
            "showing": f"first {PREVIEW_LINES} + last {PREVIEW_LINES} lines (file too large)",
            "content": head + f"\n\n... [{total - 2 * PREVIEW_LINES} lines omitted] ...\n\n" + tail,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_list_file(workspace: str, path: str = ".") -> dict:
    try:
        full = _validate_path(workspace, path)
        entries = sorted(os.listdir(full))
        return {"ok": True, "entries": entries}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_execute_phreeqc_raw(workspace: str, input_path: str) -> dict:
    """Run PHREEQC and return RAW result.out content (no TOC)."""
    try:
        full_in = _validate_path(workspace, input_path)
        if not Path(full_in).exists():
            return {"ok": False, "error": f"Input file not found: {input_path}"}

        workdir = Path(full_in).parent
        out_file = workdir / "result.out"
        files_before = set(os.listdir(workdir))

        cmd = [PHREEQC_BIN, str(full_in), str(out_file), str(DEFAULT_DB)]
        p = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)

        after = set(os.listdir(workdir))
        new_files = sorted(after - files_before)
        output_files = [f for f in new_files if (workdir / f).is_file()]

        file_sizes = {}
        for f in output_files:
            fpath = workdir / f
            if fpath.exists():
                file_sizes[f] = fpath.stat().st_size

        # KEY: return raw content instead of TOC
        raw_content = ""
        truncated = False
        if out_file.exists():
            content = out_file.read_text(encoding="utf-8", errors="replace")
            if len(content) > MAX_RAW_CHARS:
                half = MAX_RAW_CHARS // 2
                raw_content = (
                    content[:half]
                    + f"\n\n... [TRUNCATED: {len(content)} total chars, showing first and last {half}] ...\n\n"
                    + content[-half:]
                )
                truncated = True
            else:
                raw_content = content

        result = {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout[-4000:],
            "stderr": p.stderr[-4000:],
            "output_files": output_files,
            "file_sizes": file_sizes,
            "result_out_content": raw_content,
        }
        if truncated:
            result["warning"] = f"result.out truncated from {len(content)} to ~{MAX_RAW_CHARS} chars."
        return result

    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ── TOC mode (matching original agent behavior) ──

MAX_TOC_ENTRIES = 50

def _build_toc(filepath: Path):
    """Scan result.out for section delimiters and return a table of contents."""
    toc = []
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
    except Exception:
        pass
    if len(toc) > MAX_TOC_ENTRIES:
        half = MAX_TOC_ENTRIES // 2
        trimmed = toc[:half] + [{"line": -1, "section": f"... {len(toc) - MAX_TOC_ENTRIES} sections omitted ..."}] + toc[-half:]
        return trimmed
    return toc


def tool_execute_phreeqc_toc(workspace: str, input_path: str) -> dict:
    """Run PHREEQC and return TOC (matching original agent behavior)."""
    try:
        full_in = _validate_path(workspace, input_path)
        if not Path(full_in).exists():
            return {"ok": False, "error": f"Input file not found: {input_path}"}

        workdir = Path(full_in).parent
        out_file = workdir / "result.out"
        files_before = set(os.listdir(workdir))

        cmd = [PHREEQC_BIN, str(full_in), str(out_file), str(DEFAULT_DB)]
        p = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)

        after = set(os.listdir(workdir))
        new_files = sorted(after - files_before)
        output_files = [f for f in new_files if (workdir / f).is_file()]

        file_sizes = {}
        for f in output_files:
            fpath = workdir / f
            if fpath.exists():
                file_sizes[f] = fpath.stat().st_size

        toc = []
        toc_skipped = False
        if out_file.exists():
            if out_file.stat().st_size <= 50 * 1024 * 1024:
                toc = _build_toc(out_file)
            else:
                toc_skipped = True

        result = {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout[-4000:],
            "stderr": p.stderr[-4000:],
            "output_files": output_files,
            "file_sizes": file_sizes,
            "result_out_toc": toc,
            "hint": "Use read_file with start_line/end_line to read specific sections.",
        }
        if toc_skipped:
            result["warning"] = (
                f"result.out is very large ({out_file.stat().st_size / 1e6:.0f} MB). "
                "TOC scan skipped. Use read_file to inspect the tail of the file."
            )
        return result

    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ── Tool dispatch ──

# Default: raw mode. Switched to TOC mode via --toc flag in main()
TOOL_DISPATCH = {
    "write_file": lambda ws, args: tool_write_file(ws, **args),
    "read_file": lambda ws, args: tool_read_file(ws, **args),
    "list_file": lambda ws, args: tool_list_file(ws, **args),
    "execute_phreeqc": lambda ws, args: tool_execute_phreeqc_raw(ws, **args),
}

# litellm tool specs (OpenAI function-calling format; litellm translates for Anthropic)
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "new_content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file. Optionally specify start_line/end_line (1-based) for a range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "start_line": {"type": "string", "description": "Start line (1-based, optional)."},
                    "end_line": {"type": "string", "description": "End line (1-based, optional)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_file",
            "description": "List files and directories in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: workspace root)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_phreeqc",
            "description": (
                "Run PHREEQC on an input file. Returns execution status, "
                "stdout/stderr, output file paths, and result content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to the PHREEQC input file."},
                },
                "required": ["input_path"],
            },
        },
    },
]


# ── Logging ──

def _log_event(log_path: Path, action: str, **data):
    event = {"ts": datetime.now().isoformat(timespec="seconds"), "action": action, **data}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ── Agent loop ──

def _is_anthropic_model(model: str) -> bool:
    """Check if model is an Anthropic Claude model (needs step-warning to match original)."""
    return "claude" in model.lower() or "anthropic" in model.lower()


def run_agent_raw(question_prompt: str, workspace: str, model: str, max_steps: int = 24, toc_mode: bool = False) -> None:
    """ReAct agent loop using litellm. Supports raw output or TOC mode."""
    log_dir = Path(workspace) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"chat_{ts}_{int(time.time())}.jsonl"

    ablation_label = "toc" if toc_mode else "raw_output"
    _log_event(log_path, "run_start", model=model, max_steps=max_steps, ablation=ablation_label)
    _log_event(log_path, "user", content=question_prompt)

    sys_prompt = SYSTEM_PROMPT_ORIGINAL if toc_mode else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_prompt},
    ]

    use_step_warning = _is_anthropic_model(model)

    for step in range(1, max_steps + 1):
        # Step-limit warning: only for Anthropic models (matching agent_anthropic.py)
        # OpenAI agent.py does NOT have this warning
        if use_step_warning and step == max_steps - 1:
            messages.append({
                "role": "user",
                "content": (
                    "WARNING: You have only 2 steps remaining. "
                    "You MUST write final_answer.txt NOW with your best answer. "
                    "Use write_file to create final_answer.txt with content: <<< X >>> "
                    "where X is A, B, C, or D."
                ),
            })

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                # Match original settings per provider:
                # agent_anthropic.py: temperature=0, max_tokens=4096
                # agent.py (OpenAI): no explicit temperature, no explicit max_tokens
                completion_kwargs = {
                    "model": model,
                    "messages": messages,
                    "tools": TOOL_SPECS,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
                if _is_anthropic_model(model):
                    completion_kwargs["temperature"] = 0
                    completion_kwargs["max_tokens"] = 4096
                # OpenAI: use provider defaults (no temp, no max_tokens)

                response = litellm.completion(**completion_kwargs)
                break
            except Exception as api_err:
                err_str = str(api_err)
                is_transient = any(k in err_str for k in ("429", "500", "502", "503", "529", "timeout", "connection", "overloaded", "rate_limit"))
                if not is_transient or attempt == max_retries:
                    _log_event(log_path, "run_end", status="api_error", step=step, error=err_str)
                    raise
                # Rate-limit backoff: 15s, 30s, 45s, ... up to 90s
                wait = min(15 * attempt, 90)
                logger.warning("API call failed (attempt %d/%d, wait %ds): %s", attempt, max_retries, wait, api_err)
                time.sleep(wait)

        # Small delay between steps to stay under rate limit
        time.sleep(1)

        choice = response.choices[0]
        assistant_msg = choice.message
        tool_calls = assistant_msg.tool_calls or []

        # Log usage
        usage_dict = {}
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
            tool_calls=[
                {"id": first_tc.id, "name": first_tc.function.name, "arguments": first_tc.function.arguments}
            ] if first_tc else [],
        )

        # Build assistant message for context
        assistant_payload = {"role": "assistant", "content": assistant_msg.content or ""}
        if first_tc:
            assistant_payload["tool_calls"] = [{
                "id": first_tc.id,
                "type": "function",
                "function": {
                    "name": first_tc.function.name,
                    "arguments": first_tc.function.arguments or "{}",
                },
            }]
        messages.append(assistant_payload)

        # No tool call -> done
        if first_tc is None:
            _log_event(log_path, "run_end", status="answered", step=step)
            return

        # Execute tool
        tool_name = first_tc.function.name
        try:
            tool_args = json.loads(first_tc.function.arguments or "{}")
        except json.JSONDecodeError:
            tool_args = {}

        dispatch = TOOL_DISPATCH.get(tool_name)
        if dispatch is None:
            result = {"ok": False, "error": f"Unknown tool: {tool_name}"}
        else:
            try:
                result = dispatch(workspace, tool_args)
            except Exception as e:
                result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        messages.append({
            "role": "tool",
            "tool_call_id": first_tc.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

        _log_event(log_path, "tool", step=step, tool=tool_name, args=tool_args, result=result)

    _log_event(log_path, "run_end", status="max_steps_reached", step=max_steps)


# ── Evaluation harness (same structure as evaluate.py) ──

def build_prompt(question_text: str) -> str:
    return f"""Answer the multiple-choice question below.

Question:
{question_text}

Steps:
1) Use tools as needed for PHREEQC/quantitative work.
2) Your response should be in the form <<< X >>>, where X is either A, B, C, or D.
3) Use write_file to create or overwrite final_answer.txt in the current workspace.
   The file content must be exactly: <<< X >>> followed by a newline.
"""


def _parse_choice(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"<<<\s*([A-D])\s*>>>", text.strip(), flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def _run_in_process(row, result_queue, ws_root, model, max_steps, toc_mode=False):
    try:
        # IMPORTANT: On macOS (spawn), child process re-imports module and gets default
        # TOOL_DISPATCH (raw mode). Must set dispatch here inside the subprocess.
        if toc_mode:
            TOOL_DISPATCH["execute_phreeqc"] = lambda ws, a: tool_execute_phreeqc_toc(ws, **a)

        idx = row["index"]
        workspace = os.path.join(ws_root, f"dataset_q_{idx + 1:05d}")
        os.makedirs(workspace, exist_ok=True)

        db_ref = BASE_DIR / "database_reference.txt"
        if db_ref.exists():
            shutil.copy(db_ref, os.path.join(workspace, "database_reference.txt"))

        prompt = build_prompt(row["question"])
        run_agent_raw(prompt, workspace, model, max_steps, toc_mode=toc_mode)
    except Exception as e:
        result_queue.put({"error": str(e)})
        return
    result_queue.put({"ok": True})


def _process_one(row, ws_root, model, max_steps=24, toc_mode=False):
    idx = row["index"]
    truth = str(row["answer"]).strip().upper()
    if truth not in {"A", "B", "C", "D"}:
        truth = None

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run_in_process, args=(row, result_queue, ws_root, model, max_steps, toc_mode))
    proc.start()
    proc.join(timeout=600)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"index": idx + 1, "truth": truth, "prediction": None, "is_correct": None, "error": "Timeout 600s"}

    if proc.exitcode != 0:
        error_msg = "Process crashed"
        if not result_queue.empty():
            msg = result_queue.get_nowait()
            if isinstance(msg, dict) and msg.get("error"):
                error_msg = msg["error"]
        return {"index": idx + 1, "truth": truth, "prediction": None, "is_correct": None, "error": error_msg}

    ws = Path(ws_root) / f"dataset_q_{idx + 1:05d}"
    fa = ws / "final_answer.txt"
    pred = _parse_choice(fa.read_text(encoding="utf-8")) if fa.exists() else None

    return {
        "index": idx + 1,
        "truth": truth,
        "prediction": pred,
        "is_correct": (pred == truth) if pred and truth else None,
    }


def _load_dataset(path: Path):
    records = []
    with path.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q, a = row.get("question"), row.get("answer")
            if q and a is not None:
                records.append({"index": idx, "question": str(q), "answer": str(a)})
    return records


def main():
    parser = argparse.ArgumentParser(description="TOC Ablation: agent with raw PHREEQC output (no TOC)")
    parser.add_argument("--dataset", default=str(DATASET_FILE))
    parser.add_argument("--model", required=True, help="litellm model string, e.g. claude-opus-4-6 or gpt-5.2")
    parser.add_argument("--name", required=True, help="Run name for result subfolder")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Skip questions with existing final_answer.txt")
    parser.add_argument("--toc", action="store_true", help="Use TOC mode (original agent behavior) instead of raw output")
    args = parser.parse_args()

    # Switch execute_phreeqc dispatch based on mode
    if args.toc:
        TOOL_DISPATCH["execute_phreeqc"] = lambda ws, a: tool_execute_phreeqc_toc(ws, **a)
        print(f"Mode: TOC (original agent behavior, with token logging)")

    records = _load_dataset(Path(args.dataset))
    run_dir = RESULT_ROOT / "agent" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ws_root = str(run_dir / "work_space")
    os.makedirs(ws_root, exist_ok=True)

    results_list = []

    if args.resume:
        remaining = []
        for row in records:
            idx = row["index"]
            fa = Path(ws_root) / f"dataset_q_{idx + 1:05d}" / "final_answer.txt"
            if fa.exists():
                truth = str(row["answer"]).strip().upper()
                pred = _parse_choice(fa.read_text(encoding="utf-8"))
                results_list.append({
                    "index": idx + 1,
                    "truth": truth if truth in {"A","B","C","D"} else None,
                    "prediction": pred,
                    "is_correct": (pred == truth) if pred and truth in {"A","B","C","D"} else None,
                })
            else:
                remaining.append(row)
        print(f"Resuming: {len(results_list)} done, {len(remaining)} remaining")
        records = remaining

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_process_one, row, ws_root, args.model, args.max_steps, args.toc): row
            for row in records
        }
        it = tqdm(as_completed(futures), total=len(futures), desc=f"Ablation ({args.model})", unit="q") if tqdm else as_completed(futures)
        for future in it:
            results_list.append(future.result())

    results_list.sort(key=lambda r: r["index"])

    results_path = run_dir / "results.jsonl"
    with results_path.open("w") as f:
        for r in results_list:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results_list)
    correct = sum(1 for r in results_list if r.get("is_correct") is True)
    errors = sum(1 for r in results_list if r.get("error"))

    # Aggregate token usage from logs
    total_input_tokens = 0
    total_output_tokens = 0
    for qdir in Path(ws_root).glob("dataset_q_*"):
        for logf in qdir.glob("logs/chat_*.jsonl"):
            with open(logf) as lf:
                for line in lf:
                    try:
                        e = json.loads(line.strip())
                        if e.get("action") == "assistant" and e.get("usage"):
                            total_input_tokens += e["usage"].get("input_tokens", 0)
                            total_output_tokens += e["usage"].get("output_tokens", 0)
                    except:
                        pass

    summary = {
        "model": args.model,
        "method": "agent_toc" if args.toc else "agent_raw_output_no_toc",
        "ablation": "toc" if args.toc else "raw_output",
        "dataset": Path(args.dataset).name,
        "run_name": args.name,
        "total_rows": total,
        "correct_rows": correct,
        "error_rows": errors,
        "accuracy": correct / total if total else 0,
        "max_raw_chars": MAX_RAW_CHARS if not args.toc else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nResults: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
