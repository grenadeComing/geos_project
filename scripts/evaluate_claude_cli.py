"""
Claude Code CLI black-box evaluation runner.

Spawns `claude -p ...` per question, with all default Claude Code tools
(Bash, Read, Write, Edit, Glob, Grep). The agent is told where the PHREEQC
binary and database live and asked to write its final answer to
final_answer.txt. PHREEQC output is read raw via Bash/Read; this maps closest
to the custom Agent-Raw condition (no TOC scaffolding).

Usage:
  # Smoke test on the first 5 questions:
  python scripts/evaluate_claude_cli.py --model claude-sonnet-4-6 \
      --name cli_sonnet46_smoke --workers 2 --limit 5

  # Full run:
  python scripts/evaluate_claude_cli.py --model claude-opus-4-6 --name cli_opus46 --workers 4
  python scripts/evaluate_claude_cli.py --model claude-sonnet-4-6 --name cli_sonnet46 --workers 4

Requires: `claude` on PATH and either an authenticated Claude Code session
(`claude /login`) or ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from _common import (  # noqa: E402
    grade_row,
    load_dataset,
    parse_answer,
    summarize_results,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_DATASET = REPO_ROOT / "datasets" / "neurips_dataset_200_clean.jsonl"
RESULT_ROOT = REPO_ROOT / "result" / "claude_cli"
DEFAULT_MODEL = "claude-sonnet-4-6"

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
    raise FileNotFoundError(
        f"PHREEQC database not found: {DEFAULT_DB}. Set PHREEQC_DB to a valid path."
    )
DB_REFERENCE = REPO_ROOT / "archive_acm_2026" / "all_materials" / "database_reference.txt"


PHREEQC_INSTRUCTION = f"""\
You have access to the PHREEQC geochemical simulator.
Binary: {PHREEQC_BIN}
Thermodynamic database: ./phreeqc.dat (in your current working directory)
Element/species reference: ./database_reference.txt

Run via Bash: {PHREEQC_BIN} <input.pqi> <output.out> ./phreeqc.dat
"""


def _build_prompt(row: dict[str, Any]) -> str:
    if row["task_type"] == "mcq":
        answer_format = "Respond with ONLY the answer letter (A, B, C, or D). No explanation, no other text."
    else:
        answer_format = "Respond with ONLY the numeric answer. No units, no explanation, no other text."
    return f"{PHREEQC_INSTRUCTION}\n{row['question']}\n\n{answer_format}\n"


def _extract_pred(result_text: str, task_type: str) -> Any:
    """Strict single-letter / single-number first; fall back to parse_answer."""
    s = (result_text or "").strip()
    if task_type == "mcq":
        u = s.upper()
        if u in ("A", "B", "C", "D"):
            return u
    return parse_answer(s, task_type, strict=False)


def _process_one(row: dict[str, Any], ws_root: Path, model: str,
                 max_turns: int, timeout_s: int) -> dict[str, Any]:
    idx = row["index"]
    workspace = ws_root / f"dataset_q_{idx + 1:05d}"
    workspace.mkdir(parents=True, exist_ok=True)

    if not (workspace / "phreeqc.dat").exists():
        shutil.copy(DEFAULT_DB, workspace / "phreeqc.dat")
    if DB_REFERENCE.exists() and not (workspace / "database_reference.txt").exists():
        shutil.copy(DB_REFERENCE, workspace / "database_reference.txt")

    prompt = _build_prompt(row)
    log_path = workspace / "cli_log.json"

    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--max-turns", str(max_turns),
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--disallowedTools", "WebSearch,WebFetch",
    ]

    started = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log_path.write_text(json.dumps(
            {"error": f"Timeout {timeout_s}s", "elapsed_s": time.time() - started},
            indent=2,
        ))
        result = grade_row(row, pred=None)
        result["error"] = f"Timeout {timeout_s}s"
        return result

    elapsed = time.time() - started

    parsed: Any = None
    try:
        parsed = json.loads(p.stdout)
    except Exception:
        parsed = None

    log = {
        "returncode": p.returncode,
        "elapsed_s": round(elapsed, 3),
        "parsed": parsed,
        "stdout_raw_tail": p.stdout[-2000:] if parsed is None else None,
        "stderr_tail": p.stderr[-2000:] if p.stderr else "",
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))

    pred: Any = None
    if isinstance(parsed, dict) and parsed.get("result"):
        pred = _extract_pred(parsed["result"], row["task_type"])

    return grade_row(row, pred)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Claude Code CLI black-box runner.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--name", required=True, help="Run name for result subfolder.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=24)
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-question wall-clock timeout (seconds).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N questions (for smoke tests).")
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly sample N questions (use --seed for reproducibility).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for --sample. Default 42.")
    p.add_argument("--resume", action="store_true",
                   help="Skip questions whose cli_log.json already exists.")
    p.add_argument("--task-filter", choices=("all", "mcq", "fill"), default="all")
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
    if args.limit is not None:
        records = records[:args.limit]
        print(f"Limit applied: running first {len(records)} questions.")
    if args.sample is not None:
        import random
        if args.sample > len(records):
            raise RuntimeError(f"--sample {args.sample} > available {len(records)} records.")
        rng = random.Random(args.seed)
        records = sorted(rng.sample(records, args.sample), key=lambda r: r["index"])
        sampled_indices = [r["index"] for r in records]
        print(f"Sampled {len(records)} questions with seed={args.seed}: indices {sampled_indices}")

    run_dir = RESULT_ROOT / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ws_root = run_dir / "work_space"
    ws_root.mkdir(parents=True, exist_ok=True)

    if args.sample is not None:
        (run_dir / "sample_indices.json").write_text(json.dumps({
            "seed": args.seed,
            "n_sampled": len(records),
            "n_total": len(load_dataset(dataset_path)),
            "indices": [r["index"] for r in records],
        }, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    records_to_run = records

    results_path = run_dir / "results.jsonl"

    if args.resume:
        remaining: list[dict[str, Any]] = []
        for row in records:
            idx = row["index"]
            logf = ws_root / f"dataset_q_{idx + 1:05d}" / "cli_log.json"
            if logf.exists():
                try:
                    log = json.loads(logf.read_text())
                    env = log.get("parsed") if isinstance(log, dict) else None
                    pred = (_extract_pred(env["result"], row["task_type"])
                            if isinstance(env, dict) and env.get("result") else None)
                    results.append(grade_row(row, pred))
                    continue
                except Exception:
                    pass
            remaining.append(row)
        print(f"Resuming: {len(results)} done, {len(remaining)} remaining")
        records_to_run = remaining
    else:
        # Fresh run: clear streaming results.jsonl
        results_path.write_text("", encoding="utf-8")

    write_lock = threading.Lock()

    def _append_result(r: dict[str, Any]) -> None:
        with write_lock:
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()

    # Seed the streaming file with already-resumed results so it's the full record.
    for r in results:
        _append_result(r)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(_process_one, row, ws_root, args.model,
                      args.max_turns, args.timeout): row
            for row in records_to_run
        }
        it = (tqdm(as_completed(futures), total=len(futures),
                   desc=f"cli ({args.model})", unit="q")
              if tqdm else as_completed(futures))
        for fut in it:
            r = fut.result()
            results.append(r)
            _append_result(r)

    # Final canonical rewrite: streaming file is in completion order; sort by index for analysis tools.
    results.sort(key=lambda r: r["index"])
    results_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )

    agg = summarize_results(results)
    ended_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - started_perf

    agg_input = agg_output = agg_cache_read = agg_cache_create = 0
    agg_cost_usd = 0.0
    agg_duration_ms = agg_num_turns = 0
    qs_with_usage = 0

    for qdir in sorted(ws_root.glob("dataset_q_*")):
        logf = qdir / "cli_log.json"
        if not logf.exists():
            continue
        try:
            log = json.loads(logf.read_text())
        except Exception:
            continue
        env = log.get("parsed") if isinstance(log, dict) else None
        if not isinstance(env, dict):
            continue
        u = env.get("usage")
        if isinstance(u, dict):
            agg_input += int(u.get("input_tokens") or 0)
            agg_output += int(u.get("output_tokens") or 0)
            agg_cache_read += int(u.get("cache_read_input_tokens") or 0)
            agg_cache_create += int(u.get("cache_creation_input_tokens") or 0)
            qs_with_usage += 1
        c = env.get("total_cost_usd")
        if isinstance(c, (int, float)):
            agg_cost_usd += float(c)
        d = env.get("duration_ms")
        if isinstance(d, (int, float)):
            agg_duration_ms += int(d)
        nt = env.get("num_turns")
        if isinstance(nt, (int, float)):
            agg_num_turns += int(nt)

    total_input_effective = agg_input + agg_cache_read + agg_cache_create

    summary = {
        "dataset": dataset_path.name,
        "provider": "anthropic",
        "model": args.model,
        "method": "claude_code_cli",
        "context_mode": "raw_via_bash",
        "run_name": args.name,
        "workers": args.workers,
        "max_turns": args.max_turns,
        "task_filter": args.task_filter,
        "limit": args.limit,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "seconds_per_row": round(elapsed_seconds / len(records), 3) if records else None,
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
