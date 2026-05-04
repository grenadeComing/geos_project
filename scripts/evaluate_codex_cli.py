"""
OpenAI Codex CLI black-box evaluation runner.

Spawns `codex exec ...` per question with sandbox bypassed and approval
policy `never`. The agent has Codex's full default toolset (Bash, Read,
Write, etc.) and is told where the PHREEQC binary and database live.
PHREEQC output is read raw via Bash; this maps closest to the custom
Agent-Raw condition.

Usage:
  # Smoke test on first question:
  python scripts/evaluate_codex_cli.py --model gpt-5.4 \
      --name codex_gpt54_smoke --workers 1 --limit 1

  # Matched 50-q comparison (same indices as Sonnet SDK):
  python scripts/evaluate_codex_cli.py --model gpt-5.4 \
      --name codex_gpt54_n50 --workers 2 --sample 50 --seed 42

Requires: `codex` on PATH and either an authenticated session
(`codex login`) or `OPENAI_API_KEY` in the environment.
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
RESULT_ROOT = REPO_ROOT / "result" / "codex_cli"
DEFAULT_MODEL = "gpt-5.4"

CODEX_BIN = os.environ.get("CODEX_BIN", "/opt/homebrew/bin/codex")

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

Run via shell: {PHREEQC_BIN} <input.pqi> <output.out> ./phreeqc.dat
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
                 timeout_s: int) -> dict[str, Any]:
    idx = row["index"]
    workspace = ws_root / f"dataset_q_{idx + 1:05d}"
    workspace.mkdir(parents=True, exist_ok=True)

    if not (workspace / "phreeqc.dat").exists():
        shutil.copy(DEFAULT_DB, workspace / "phreeqc.dat")
    if DB_REFERENCE.exists() and not (workspace / "database_reference.txt").exists():
        shutil.copy(DB_REFERENCE, workspace / "database_reference.txt")

    prompt = _build_prompt(row)
    log_path = workspace / "codex_log.json"
    last_msg_path = workspace / "codex_last.txt"

    cmd = [
        CODEX_BIN, "exec",
        "--model", model,
        "--cd", str(workspace),
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-c", "model_reasoning_effort=medium",
        "--json",
        "--output-last-message", str(last_msg_path),
        prompt,
    ]

    started = time.time()
    try:
        p = subprocess.run(
            cmd,
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

    # Parse JSONL events from stdout to extract usage if possible
    events = []
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            pass

    # Compute aggregate usage from events (Codex emits token usage events)
    total_input = total_output = 0
    n_turns = 0
    for ev in events:
        # Best-effort extraction; field names vary by codex version
        for k in ("input_tokens", "prompt_tokens"):
            v = ev.get(k) if isinstance(ev, dict) else None
            if isinstance(v, int):
                total_input += v
        for k in ("output_tokens", "completion_tokens"):
            v = ev.get(k) if isinstance(ev, dict) else None
            if isinstance(v, int):
                total_output += v
        # Count assistant turns
        et = ev.get("type") if isinstance(ev, dict) else None
        if et in ("assistant_message", "agent_message", "message", "response"):
            n_turns += 1

    log = {
        "returncode": p.returncode,
        "elapsed_s": round(elapsed, 3),
        "n_events": len(events),
        "n_turns_approx": n_turns,
        "total_input_tokens_approx": total_input,
        "total_output_tokens_approx": total_output,
        "stderr_tail": (p.stderr or "")[-2000:],
        "stdout_tail": (p.stdout or "")[-2000:],
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))

    # Read final message
    pred: Any = None
    if last_msg_path.exists():
        text = last_msg_path.read_text(encoding="utf-8")
        pred = _extract_pred(text, row["task_type"])

    return grade_row(row, pred)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI Codex CLI black-box runner.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--name", required=True, help="Run name for result subfolder.")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-question wall-clock timeout (seconds).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N questions (for smoke tests).")
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly sample N questions (use --seed for reproducibility).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for --sample. Default 42.")
    p.add_argument("--resume", action="store_true",
                   help="Skip questions whose codex_log.json already exists.")
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
            logf = ws_root / f"dataset_q_{idx + 1:05d}" / "codex_log.json"
            last_msg = ws_root / f"dataset_q_{idx + 1:05d}" / "codex_last.txt"
            if logf.exists():
                pred = None
                if last_msg.exists():
                    pred = _extract_pred(last_msg.read_text(encoding="utf-8"), row["task_type"])
                results.append(grade_row(row, pred))
                continue
            remaining.append(row)
        print(f"Resuming: {len(results)} done, {len(remaining)} remaining")
        records_to_run = remaining
    else:
        results_path.write_text("", encoding="utf-8")

    write_lock = threading.Lock()

    def _append_result(r: dict[str, Any]) -> None:
        with write_lock:
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()

    for r in results:
        _append_result(r)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(_process_one, row, ws_root, args.model, args.timeout): row
            for row in records_to_run
        }
        it = (tqdm(as_completed(futures), total=len(futures),
                   desc=f"codex ({args.model})", unit="q")
              if tqdm else as_completed(futures))
        for fut in it:
            r = fut.result()
            results.append(r)
            _append_result(r)

    results.sort(key=lambda r: r["index"])
    results_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )

    agg = summarize_results(results)
    ended_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - started_perf

    # Aggregate token counts from per-question codex_log.json (best-effort)
    agg_input_tokens = agg_output_tokens = agg_turns = 0
    qs_with_usage = 0
    for qdir in sorted(ws_root.glob("dataset_q_*")):
        logf = qdir / "codex_log.json"
        if not logf.exists():
            continue
        try:
            log = json.loads(logf.read_text())
        except Exception:
            continue
        i = log.get("total_input_tokens_approx") or 0
        o = log.get("total_output_tokens_approx") or 0
        t = log.get("n_turns_approx") or 0
        if i or o:
            qs_with_usage += 1
        agg_input_tokens += int(i)
        agg_output_tokens += int(o)
        agg_turns += int(t)

    summary = {
        "dataset": dataset_path.name,
        "provider": "openai",
        "model": args.model,
        "method": "codex_cli",
        "context_mode": "raw_via_bash",
        "run_name": args.name,
        "workers": args.workers,
        "task_filter": args.task_filter,
        "limit": args.limit,
        "sample": args.sample,
        "seed": args.seed if args.sample is not None else None,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "seconds_per_row": round(elapsed_seconds / len(records), 3) if records else None,
        **agg,
        "questions_with_usage": qs_with_usage,
        "total_input_tokens_approx": agg_input_tokens,
        "total_output_tokens_approx": agg_output_tokens,
        "total_turns_approx": agg_turns,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results saved to: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
