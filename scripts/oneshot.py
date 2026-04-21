"""
One-shot baseline: single LLM call per question, no tools, no agent loop.

Unified across providers via litellm:
  - Anthropic: --model claude-opus-4-6 / claude-sonnet-4-6
  - OpenAI:    --model gpt-5.1 / gpt-5.2
  - Google:    --model gemini/gemini-2.5-pro / gemini/gemini-2.5-flash

Supports both MCQ and fill-in-the-blank via the dataset's `task_type` field
(see scripts/_common.py for schema). A single run can mix task types;
per-type breakdown appears in summary.json.

Usage:
  python scripts/oneshot.py --model claude-opus-4-6 --name oneshot_opus46
  python scripts/oneshot.py --model gpt-5.2 --name oneshot_gpt52 --workers 8
  python scripts/oneshot.py --model gemini/gemini-2.5-pro --name oneshot_gem25pro

Requires: pip install litellm
Requires: ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY as appropriate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import litellm

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from _common import (  # noqa: E402
    grade_row,
    load_dataset,
    parse_answer,
    summarize_results,
    system_prompt_oneshot,
    build_prompt_oneshot,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_DATASET = REPO_ROOT / "datasets" / "phreeqc_bench_v2.jsonl"
RESULT_ROOT = REPO_ROOT / "result" / "oneshot"

TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "529",
    "timeout", "connection", "overloaded", "rate_limit",
    "resource_exhausted",  # Gemini quota (original baseline_google.py)
)


def _provider_of(model: str) -> str:
    m = model.lower()
    if m.startswith("claude") or m.startswith("anthropic/"):
        return "anthropic"
    if m.startswith("gemini") or m.startswith("gemini/") or m.startswith("google/"):
        return "google"
    return "openai"


def _completion_kwargs(model: str, system: str, user: str) -> dict[str, Any]:
    """Pick provider-appropriate call args. Matches original baseline scripts."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    provider = _provider_of(model)
    if provider == "anthropic":
        kwargs["temperature"] = 0
        kwargs["max_tokens"] = 2048
    elif provider == "google":
        # Gemini's baseline script used temperature=0 implicitly via default=0.
        kwargs["temperature"] = 0
    # OpenAI GPT-5.x reasoners: leave at provider defaults (no explicit temp/max_tokens).
    return kwargs


def _query_model(model: str, system: str, user: str, *, max_attempts: int = 5) -> tuple[str | None, dict[str, int]]:
    """Call litellm with retries. Returns (text, usage_dict)."""
    kwargs = _completion_kwargs(model, system, user)
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = litellm.completion(**kwargs)
            msg = resp.choices[0].message
            text = msg.content if hasattr(msg, "content") else None
            usage: dict[str, int] = {}
            if getattr(resp, "usage", None):
                usage = {
                    "input_tokens": int(resp.usage.prompt_tokens or 0),
                    "output_tokens": int(resp.usage.completion_tokens or 0),
                }
            return text, usage
        except Exception as e:  # noqa: BLE001
            last_err = e
            if not any(k in str(e).lower() for k in TRANSIENT_MARKERS) or attempt == max_attempts:
                raise
            time.sleep(min(2 ** attempt, 30))
    if last_err:
        raise last_err
    return None, {}


def _process_one(row: dict[str, Any], model: str) -> dict[str, Any]:
    task_type = row["task_type"]
    system = system_prompt_oneshot(task_type)
    user = build_prompt_oneshot(row)

    try:
        raw, usage = _query_model(model, system, user)
    except Exception as e:  # noqa: BLE001
        result = grade_row(row, pred=None)
        result.update({"error": f"{type(e).__name__}: {e}", "usage": {}})
        return result

    pred = parse_answer(raw, task_type)
    result = grade_row(row, pred)
    result["raw_response"] = raw
    result["usage"] = usage
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-shot baseline (litellm). MCQ + fill.")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to JSONL dataset.")
    p.add_argument("--model", required=True, help="litellm model string.")
    p.add_argument("--name", required=True, help="Run name for result subfolder.")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers (threads).")
    p.add_argument(
        "--task-filter", choices=("all", "mcq", "fill"), default="all",
        help="Only run a subset of questions by task_type.",
    )
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

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(_process_one, row, args.model): row for row in records}
        it = (
            tqdm(as_completed(futures), total=len(futures), desc=f"oneshot ({args.model})", unit="q")
            if tqdm else as_completed(futures)
        )
        for fut in it:
            results.append(fut.result())

    results.sort(key=lambda r: r["index"])
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )

    agg = summarize_results(results)
    total_input = sum((r.get("usage") or {}).get("input_tokens", 0) for r in results)
    total_output = sum((r.get("usage") or {}).get("output_tokens", 0) for r in results)

    summary = {
        "provider": _provider_of(args.model),
        "model": args.model,
        "method": "one_shot_no_tools",
        "dataset": dataset_path.name,
        "run_name": args.name,
        "task_filter": args.task_filter,
        "workers": args.workers,
        **agg,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results saved to: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
