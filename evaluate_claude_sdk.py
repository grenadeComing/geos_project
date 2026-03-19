# python evaluate_claude_sdk.py --dataset dataset_S+J.jsonl --model claude-opus-4-6 --name sdk_opus46_v1 --workers 4
"""
Evaluate the Claude Agent SDK on PHREEQC MCQ benchmarks.

The SDK agent gets:
  - The question as a prompt
  - The execute_phreeqc MCP tool (same tool the custom agents use)
  - Built-in tools (Read, Write, Bash, Glob, Grep) — SDK decides how to use them
  - No custom system prompt — the SDK figures things out on its own

This is a standalone evaluator parallel to evaluate.py (custom agent).
"""
import argparse
import asyncio
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
AGENT_DIR = BASE_DIR / "agent"
sys.path.insert(0, str(AGENT_DIR))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DATASET_FILE = BASE_DIR / "dataset_S+J.jsonl"
RESULT_ROOT = BASE_DIR / "result"

DEFAULT_MODEL = "claude-sonnet-4-6"


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_choice(text: str | None) -> str | None:
    if not text:
        return None
    wrapped = re.search(r"<<<\s*([A-D])\s*>>>", text.strip(), flags=re.IGNORECASE)
    if wrapped:
        return wrapped.group(1).upper()
    return None


def _build_prompt(question_text: str, workspace: str) -> str:
    return f"""Answer the multiple-choice question below.

Your working directory is: {workspace}
All files must be read from and written to this directory only.

Question:
{question_text}

Write your final answer as <<< X >>> (where X is A, B, C, or D) in final_answer.txt.
"""


# ── MCP tool: expose execute_phreeqc to the SDK ─────────────────────────

def _build_mcp_server(workspace: str):
    from claude_agent_sdk import tool, create_sdk_mcp_server
    from tools.base_tool import BaseTool
    from tools.executation_tool import ExecutePHREEQCTool

    BaseTool.allowed_root = workspace
    executor = ExecutePHREEQCTool()

    @tool(
        "execute_phreeqc",
        "Run PHREEQC geochemical simulation. "
        "input_path must be a file path (e.g. 'input.pqi') to a .pqi file "
        "in the workspace, NOT the file content. "
        "Write the .pqi file first, then pass its path here.",
        {"input_path": str},
    )
    async def mcp_execute(args):
        result = executor.run(input_path=args["input_path"])
        # Trim response for MCP transport stability (keep full data in custom agent)
        compact = {
            "ok": result.get("ok"),
            "returncode": result.get("returncode"),
            "stderr": (result.get("stderr") or "")[-500:],
            "output_files": result.get("output_files", []),
            "result_out_toc": result.get("result_out_toc", []),
            "hint": "Use Read tool with offset/limit to read specific sections of result.out.",
        }
        return {"content": [{"type": "text", "text": json.dumps(compact, ensure_ascii=False)}]}

    return create_sdk_mcp_server(
        name="phreeqc-tools",
        version="1.0.0",
        tools=[mcp_execute],
    )


# ── Logging ──────────────────────────────────────────────────────────────

def _log_event(log_path: Path, event: dict):
    try:
        event.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _make_hooks(log_path: Path):
    from claude_agent_sdk import HookMatcher

    async def pre_hook(input_data, tool_use_id, context):
        _log_event(log_path, {
            "event": "pre_tool_use",
            "tool": input_data.get("tool_name", "unknown"),
            "tool_use_id": tool_use_id,
            "arguments": input_data.get("tool_input", {}),
        })
        return {}

    async def post_hook(input_data, tool_use_id, context):
        _log_event(log_path, {
            "event": "post_tool_use",
            "tool": input_data.get("tool_name", "unknown"),
            "tool_use_id": tool_use_id,
        })
        return {}

    return {
        "PreToolUse": [HookMatcher(matcher=".*", hooks=[pre_hook])],
        "PostToolUse": [HookMatcher(matcher=".*", hooks=[post_hook])],
    }


def _log_message(log_path: Path, message):
    try:
        if hasattr(message, "result"):
            _log_event(log_path, {"event": "result", "result": message.result or ""})
            return
        raw = None
        if hasattr(message, "data") and isinstance(message.data, dict):
            raw = message.data
        elif hasattr(message, "__dict__"):
            raw = {}
            for k, v in vars(message).items():
                try:
                    json.dumps(v)
                    raw[k] = v
                except (TypeError, ValueError):
                    raw[k] = str(v)[:500]
        _log_event(log_path, {
            "event": "sdk_message",
            "type": getattr(message, "type", getattr(message, "subtype", type(message).__name__)),
            "data": raw or str(message)[:1000],
        })
    except Exception:
        pass


# ── SDK runner (async) ───────────────────────────────────────────────────

async def _run_sdk_async(question_text: str, workspace: str, model: str, log_path: Path) -> str:
    from claude_agent_sdk import query, ClaudeAgentOptions

    prompt = _build_prompt(question_text, workspace)
    mcp_server = _build_mcp_server(workspace)

    _log_event(log_path, {"event": "run_start", "model": model})

    final_text = ""
    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=workspace,
                model=model,
                permission_mode="bypassPermissions",
                max_turns=24,
                mcp_servers={"phreeqc-tools": mcp_server},
                setting_sources=[],
            ),
        ):
            _log_message(log_path, message)
            if hasattr(message, "result"):
                final_text = message.result or ""
    except Exception as e:
        _log_event(log_path, {"event": "run_end", "status": "error", "error": f"{type(e).__name__}: {e}"})
        raise

    _log_event(log_path, {"event": "run_end", "status": "completed"})
    return final_text


def _run_sdk_sync(question_text: str, workspace: str, model: str, log_path: Path) -> str:
    return asyncio.run(_run_sdk_async(question_text, workspace, model, log_path))


# ── Process wrapper (isolated per question) ──────────────────────────────

def _run_in_process(row: dict, result_queue: multiprocessing.Queue, ws_root: str, model: str) -> None:
    try:
        idx = row["index"]
        question_text = row["question"]

        workspace_name = f"dataset_q_{idx + 1:05d}"
        question_workspace = Path(ws_root) / workspace_name
        question_workspace.mkdir(parents=True, exist_ok=True)

        # Copy database reference
        db_ref = BASE_DIR / "database_reference.txt"
        if db_ref.exists():
            shutil.copy(db_ref, question_workspace / "database_reference.txt")

        log_path = question_workspace / "sdk_log.jsonl"
        _run_sdk_sync(question_text, str(question_workspace), model, log_path)
    except Exception as e:
        result_queue.put({"error": str(e)})
        return

    result_queue.put({"ok": True})


def _process_one_question(row: dict, ws_root: str, model: str) -> dict:
    idx = row["index"]
    truth_raw = row["answer"]
    truth = str(truth_raw).strip().upper()
    if truth not in {"A", "B", "C", "D"}:
        truth = None

    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run_in_process, args=(row, result_queue, ws_root, model))
    proc.start()
    proc.join(timeout=600)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {
            "index": idx + 1,
            "truth": truth,
            "prediction": None,
            "is_correct": None,
            "error": "Timed out after 600 seconds",
        }

    if proc.exitcode != 0:
        error_msg = "Process crashed"
        if not result_queue.empty():
            msg = result_queue.get_nowait()
            if isinstance(msg, dict) and msg.get("error"):
                error_msg = msg["error"]
        return {
            "index": idx + 1,
            "truth": truth,
            "prediction": None,
            "is_correct": None,
            "error": f"{error_msg} (exit code {proc.exitcode})",
        }

    question_workspace = Path(ws_root) / f"dataset_q_{idx + 1:05d}"
    final_answer_path = question_workspace / "final_answer.txt"
    final_answer_text = (
        final_answer_path.read_text(encoding="utf-8")
        if final_answer_path.exists()
        else ""
    )
    pred = _parse_choice(final_answer_text)

    return {
        "index": idx + 1,
        "truth": truth,
        "prediction": pred,
        "is_correct": (pred == truth) if pred is not None and truth is not None else None,
    }


# ── Dataset loading ──────────────────────────────────────────────────────

def _load_dataset(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset file not found: {dataset_path}")

    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question")
            answer = row.get("answer")
            if not question or answer is None:
                continue
            records.append({"index": idx, "question": str(question), "answer": str(answer)})

    if not records:
        raise RuntimeError(f"No usable rows found in {dataset_path}.")
    return records


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Claude Agent SDK on PHREEQC MCQ benchmarks."
    )
    parser.add_argument("--dataset", default=str(DATASET_FILE), help="Path to dataset JSONL file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    parser.add_argument("--name", default=None, help="Run name for result subfolder")
    parser.add_argument("--resume", action="store_true", help="Skip questions with existing final_answer.txt")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = _load_dataset(Path(args.dataset))

    run_name = args.name or f"claude_sdk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RESULT_ROOT / "claude_sdk" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ws_root = run_dir / "work_space"
    ws_root.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    workers = max(1, args.workers)
    results_list: list[dict] = []

    records_to_run = records
    if args.resume:
        skipped = 0
        remaining = []
        for row in records:
            idx = row["index"]
            truth_raw = row["answer"]
            truth = str(truth_raw).strip().upper()
            if truth not in {"A", "B", "C", "D"}:
                truth = None
            fa_path = ws_root / f"dataset_q_{idx + 1:05d}" / "final_answer.txt"
            if fa_path.exists():
                pred = _parse_choice(fa_path.read_text(encoding="utf-8"))
                results_list.append({
                    "index": idx + 1,
                    "truth": truth,
                    "prediction": pred,
                    "is_correct": (pred == truth) if pred is not None and truth is not None else None,
                })
                skipped += 1
            else:
                remaining.append(row)
        records_to_run = remaining
        print(f"Resuming: {skipped} already done, {len(remaining)} remaining.")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_one_question, row, str(ws_root), args.model): row
            for row in records_to_run
        }
        iterator = (
            tqdm(as_completed(futures), total=len(futures), desc="Claude SDK eval", unit="q")
            if tqdm is not None
            else as_completed(futures)
        )
        for future in iterator:
            result_row = future.result()
            results_list.append(result_row)

    results_list.sort(key=lambda r: r["index"])
    with results_path.open("w", encoding="utf-8") as out:
        for result_row in results_list:
            out.write(json.dumps(result_row, ensure_ascii=False) + "\n")

    total = len(results_list)
    correct = sum(1 for r in results_list if r.get("is_correct") is True)
    accuracy = (correct / total) if total else 0.0
    summary = {
        "dataset": str(Path(args.dataset).name),
        "provider": "anthropic",
        "model": args.model,
        "method": "claude_sdk",
        "run_name": run_name,
        "workers": workers,
        "total_rows": total,
        "correct_rows": correct,
        "accuracy": accuracy,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results saved to: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
