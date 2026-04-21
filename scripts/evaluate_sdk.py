"""
Claude Agent SDK evaluation runner.

The SDK agent gets:
  - The question as a user prompt (with final_answer.txt format instructions)
  - An MCP tool: execute_phreeqc (TOC mode — same as the paper's custom agent)
  - Built-in SDK tools (Read, Write, Bash, Glob, Grep) chosen by the SDK itself
  - No custom system prompt — the SDK's own behavior

Context management is NOT swept here — per the project design, the SDK line is
a single point of comparison against our custom agent's context-mode sweep.
Execute_phreeqc always returns TOC metadata (matching our default agent behavior).

Supports MCQ + fill-in-the-blank via dataset `task_type` field.

Usage:
  python scripts/evaluate_sdk.py --model claude-opus-4-6 --name sdk_opus46 --workers 4
  python scripts/evaluate_sdk.py --model claude-sonnet-4-6 --name sdk_sonnet46 --workers 4

Requires: pip install claude-agent-sdk
Requires: ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from _common import (  # noqa: E402
    FILL_FINAL_ANSWER_INSTRUCTION,
    MCQ_FINAL_ANSWER_INSTRUCTION,
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


DEFAULT_DATASET = REPO_ROOT / "datasets" / "phreeqc_bench_v2.jsonl"
RESULT_ROOT = REPO_ROOT / "result" / "sdk"
DEFAULT_MODEL = "claude-sonnet-4-6"

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

MAX_TOC_ENTRIES = 50


# ── PHREEQC + TOC (inlined so this script is self-contained) ────────────

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


def _run_phreeqc(workspace: str, input_path: str) -> dict[str, Any]:
    """Run PHREEQC and return a TOC-mode result payload."""
    ws = Path(workspace).resolve()
    full_in = (ws / input_path).resolve()
    if ws not in (full_in, *full_in.parents):
        return {"ok": False, "error": f"Path escapes workspace: {input_path}"}
    if not full_in.exists():
        return {"ok": False, "error": f"Input file not found: {input_path}"}

    workdir = full_in.parent
    out_file = workdir / "result.out"
    files_before = set(os.listdir(workdir))

    cmd = [PHREEQC_BIN, str(full_in), str(out_file), str(DEFAULT_DB)]
    p = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)

    files_after = set(os.listdir(workdir))
    output_files = sorted(f for f in (files_after - files_before)
                          if (workdir / f).is_file())

    toc: list[dict[str, Any]] = []
    if out_file.exists() and out_file.stat().st_size <= 50 * 1024 * 1024:
        toc = _build_toc(out_file)

    return {
        "ok": p.returncode == 0,
        "returncode": p.returncode,
        "stderr": (p.stderr or "")[-500:],
        "output_files": output_files,
        "result_out_toc": toc,
        "hint": "Use the Read tool with offset/limit to read specific sections of result.out.",
    }


# ── MCP server (TOC-mode execute_phreeqc) ───────────────────────────────

def _build_mcp_server(workspace: str):
    from claude_agent_sdk import tool, create_sdk_mcp_server

    @tool(
        "execute_phreeqc",
        ("Run PHREEQC geochemical simulation. Returns metadata and a TOC "
         "(section line numbers) for result.out. File contents are NOT included — "
         "use the Read tool with offset/limit to inspect sections. "
         "input_path must be a relative path (e.g. 'input.pqi') to a .pqi file "
         "in the workspace, NOT the file content. "
         "Write the .pqi file first, then pass its path here."),
        {"input_path": str},
    )
    async def mcp_execute(args):
        compact = _run_phreeqc(workspace, args["input_path"])
        return {"content": [{"type": "text", "text": json.dumps(compact, ensure_ascii=False)}]}

    return create_sdk_mcp_server(name="phreeqc-tools", version="1.0.0", tools=[mcp_execute])


# ── Logging ─────────────────────────────────────────────────────────────

def _log_event(log_path: Path, event: dict) -> None:
    try:
        event.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _log_message(log_path: Path, message) -> None:
    try:
        if hasattr(message, "result"):
            usage = getattr(message, "usage", None)
            usage_dict: dict[str, Any] | str | None
            if isinstance(usage, dict):
                usage_dict = usage
            elif usage is not None:
                try:
                    usage_dict = {k: getattr(usage, k, None) for k in (
                        "input_tokens", "output_tokens",
                        "cache_creation_input_tokens", "cache_read_input_tokens",
                    )}
                except Exception:  # noqa: BLE001
                    usage_dict = str(usage)[:500]
            else:
                usage_dict = None
            _log_event(log_path, {
                "event": "result",
                "result": message.result or "",
                "usage": usage_dict,
                "total_cost_usd": getattr(message, "total_cost_usd", None),
                "duration_ms": getattr(message, "duration_ms", None),
                "duration_api_ms": getattr(message, "duration_api_ms", None),
                "num_turns": getattr(message, "num_turns", None),
                "session_id": getattr(message, "session_id", None),
                "is_error": getattr(message, "is_error", None),
                "subtype": getattr(message, "subtype", None),
            })
            return

        raw: dict[str, Any] | str | None = None
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
    except Exception:  # noqa: BLE001
        pass


# ── SDK runner ──────────────────────────────────────────────────────────

def _build_user_prompt(row: dict[str, Any], workspace: str) -> str:
    """Compose SDK user prompt: common prompt + final_answer format instruction + cwd pin."""
    core = build_prompt(row, for_agent=True)
    final = (MCQ_FINAL_ANSWER_INSTRUCTION if row["task_type"] == "mcq"
             else FILL_FINAL_ANSWER_INSTRUCTION)
    return (
        f"Your working directory is: {workspace}\n"
        "All files must be read from and written to this directory only.\n\n"
        f"{core}\n"
        f"{final}\n"
    )


async def _run_sdk_async(row: dict[str, Any], workspace: str, model: str,
                         log_path: Path, max_turns: int) -> str:
    from claude_agent_sdk import query, ClaudeAgentOptions

    prompt = _build_user_prompt(row, workspace)
    mcp_server = _build_mcp_server(workspace)

    _log_event(log_path, {"event": "run_start", "model": model,
                          "task_type": row["task_type"], "context_mode": "toc"})

    final_text = ""
    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=workspace,
                model=model,
                permission_mode="bypassPermissions",
                max_turns=max_turns,
                mcp_servers={"phreeqc-tools": mcp_server},
                setting_sources=[],
            ),
        ):
            _log_message(log_path, message)
            if hasattr(message, "result"):
                final_text = message.result or ""
    except Exception as e:  # noqa: BLE001
        _log_event(log_path, {"event": "run_end", "status": "error",
                              "error": f"{type(e).__name__}: {e}"})
        raise

    _log_event(log_path, {"event": "run_end", "status": "completed"})
    return final_text


def _run_sdk_sync(row: dict[str, Any], workspace: str, model: str,
                  log_path: Path, max_turns: int) -> str:
    return asyncio.run(_run_sdk_async(row, workspace, model, log_path, max_turns))


# ── Per-question subprocess ─────────────────────────────────────────────

def _run_in_process(row: dict[str, Any], result_queue: multiprocessing.Queue,
                    ws_root: str, model: str, max_turns: int) -> None:
    try:
        idx = row["index"]
        workspace = Path(ws_root) / f"dataset_q_{idx + 1:05d}"
        workspace.mkdir(parents=True, exist_ok=True)

        if DB_REFERENCE.exists():
            shutil.copy(DB_REFERENCE, workspace / "database_reference.txt")

        log_path = workspace / "sdk_log.jsonl"
        _run_sdk_sync(row, str(workspace), model, log_path, max_turns)
    except Exception as e:  # noqa: BLE001
        result_queue.put({"error": str(e)})
        return
    result_queue.put({"ok": True})


def _process_one(row: dict[str, Any], ws_root: str, model: str,
                 max_turns: int, timeout_s: int) -> dict[str, Any]:
    idx = row["index"]
    q: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_in_process, args=(row, q, ws_root, model, max_turns))
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
        result["error"] = f"{err} (exit code {proc.exitcode})"
        return result

    fa = Path(ws_root) / f"dataset_q_{idx + 1:05d}" / "final_answer.txt"
    pred = parse_answer(fa.read_text(encoding="utf-8"), row["task_type"], strict=True) if fa.exists() else None
    return grade_row(row, pred)


# ── CLI ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Claude Agent SDK runner (TOC mode, MCQ+fill).")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--name", required=True, help="Run name for result subfolder.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=24)
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-question wall-clock timeout (seconds).")
    p.add_argument("--resume", action="store_true",
                   help="Skip questions with an existing final_answer.txt.")
    p.add_argument("--task-filter", choices=("all", "mcq", "fill"), default="all")
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

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(_process_one, row, str(ws_root), args.model,
                      args.max_turns, args.timeout): row
            for row in records_to_run
        }
        it = (tqdm(as_completed(futures), total=len(futures),
                   desc=f"sdk ({args.model})", unit="q")
              if tqdm else as_completed(futures))
        for fut in it:
            results.append(fut.result())

    results.sort(key=lambda r: r["index"])
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )

    agg = summarize_results(results)

    # Aggregate SDK usage/cost from each question's sdk_log.jsonl result event.
    agg_input = agg_output = agg_cache_read = agg_cache_create = 0
    agg_cost_usd = 0.0
    agg_duration_ms = agg_num_turns = 0
    qs_with_usage = 0
    for qdir in sorted(ws_root.glob("dataset_q_*")):
        logf = qdir / "sdk_log.jsonl"
        if not logf.exists():
            continue
        with logf.open() as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if ev.get("event") != "result":
                    continue
                u = ev.get("usage")
                if isinstance(u, dict):
                    agg_input += int(u.get("input_tokens") or 0)
                    agg_output += int(u.get("output_tokens") or 0)
                    agg_cache_read += int(u.get("cache_read_input_tokens") or 0)
                    agg_cache_create += int(u.get("cache_creation_input_tokens") or 0)
                    qs_with_usage += 1
                c = ev.get("total_cost_usd")
                if isinstance(c, (int, float)):
                    agg_cost_usd += float(c)
                d = ev.get("duration_ms")
                if isinstance(d, (int, float)):
                    agg_duration_ms += int(d)
                nt = ev.get("num_turns")
                if isinstance(nt, (int, float)):
                    agg_num_turns += int(nt)

    # Apples-to-apples with the custom agent (which does not use prompt caching
    # via litellm): effective input tokens = billed input + cache_read + cache_create.
    total_input_effective = agg_input + agg_cache_read + agg_cache_create

    summary = {
        "dataset": dataset_path.name,
        "provider": "anthropic",
        "model": args.model,
        "method": "claude_sdk",
        "context_mode": "toc",
        "run_name": args.name,
        "workers": args.workers,
        "max_turns": args.max_turns,
        "task_filter": args.task_filter,
        **agg,
        "questions_with_usage": qs_with_usage,
        "total_input_tokens": agg_input,
        "total_cache_read_input_tokens": agg_cache_read,
        "total_cache_creation_input_tokens": agg_cache_create,
        "total_input_tokens_effective": total_input_effective,
        "total_output_tokens": agg_output,
        "total_cost_usd": round(agg_cost_usd, 4),
        "total_duration_ms": agg_duration_ms,
        "total_num_turns": agg_num_turns,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results saved to: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
