"""
Aggregate TOC vs Raw comparison across the custom agent and the Claude Agent SDK.

Walks:
  result/agent/          — custom agent runs (from evaluate.py / evaluate_ablation_raw.py)
  result/claude_sdk/     — Claude Agent SDK runs (from evaluate_claude_sdk.py)

Reads each run's summary.json, normalizes fields, and prints a comparison table.
Handles the case where older SDK runs were logged without usage fields (pre-patch).

Usage:
  python aggregate_sdk_comparison.py
  python aggregate_sdk_comparison.py --md comparison.md   # also write markdown
  python aggregate_sdk_comparison.py --only-sj            # restrict to dataset_S+J.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
RESULT_ROOT = BASE_DIR / "result"

# Human-friendly model labels
MODEL_LABELS = {
    "claude-opus-4-6": "Opus 4.6",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.4": "GPT-5.4",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}


def _load_summary(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _classify(summary: dict) -> tuple[str, str]:
    """Return (framework, mode) labels.

    framework: 'agent' (custom) or 'sdk' (Claude Agent SDK)
    mode: 'toc' or 'raw' (or 'unknown' if neither field is set)
    """
    method = summary.get("method", "")
    if method == "claude_sdk":
        framework = "sdk"
    else:
        framework = "agent"

    mode = summary.get("mode") or summary.get("ablation")
    if mode == "raw_output":
        mode = "raw"
    if mode not in ("toc", "raw"):
        # Fall back on run name heuristics
        run_name = (summary.get("run_name") or "").lower()
        if "raw" in run_name or "ablation_raw" in run_name:
            mode = "raw"
        elif "toc" in run_name:
            mode = "toc"
        else:
            mode = "toc"  # custom agent default
    return framework, mode


def _collect() -> list[dict]:
    rows = []
    for run_dir in sorted((RESULT_ROOT).rglob("summary.json")):
        # Skip anything outside agent/ or claude_sdk/
        rel = run_dir.relative_to(RESULT_ROOT)
        top = rel.parts[0] if rel.parts else ""
        if top not in ("agent", "claude_sdk"):
            continue
        s = _load_summary(run_dir)
        if not s:
            continue
        framework, mode = _classify(s)
        # Effective input tokens = what the model actually processes (no caching).
        # For SDK: input_tokens + cache_read + cache_create.
        # For custom agent: total_input_tokens (no caching used).
        cache_read = s.get("total_cache_read_input_tokens") or 0
        cache_create = s.get("total_cache_creation_input_tokens") or 0
        billed_input = s.get("total_input_tokens") or 0
        if framework == "sdk":
            effective_input = s.get("total_input_tokens_effective")
            if effective_input is None:
                effective_input = billed_input + cache_read + cache_create
        else:
            effective_input = billed_input

        rows.append({
            "run_name": s.get("run_name", run_dir.parent.name),
            "framework": framework,
            "mode": mode,
            "model": s.get("model", "?"),
            "dataset": s.get("dataset", "?"),
            "total_rows": s.get("total_rows", 0),
            "correct_rows": s.get("correct_rows", 0),
            "accuracy": s.get("accuracy"),
            "input_tokens": effective_input or None,
            "input_billed": billed_input or None,
            "output_tokens": s.get("total_output_tokens"),
            "cache_read": cache_read or None,
            "cache_create": cache_create or None,
            "cost_usd": s.get("total_cost_usd"),
            "num_turns": s.get("total_num_turns"),
            "path": str(run_dir.parent.relative_to(BASE_DIR)),
        })
    return rows


def _fmt_tokens(n: Any) -> str:
    if n is None or n == 0:
        return "–"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_acc(a: Any) -> str:
    if a is None:
        return "–"
    return f"{float(a) * 100:.1f}%"


def _fmt_cost(c: Any) -> str:
    if c is None or c == 0:
        return "–"
    return f"${float(c):.2f}"


def _table_markdown(rows: list[dict]) -> str:
    header = [
        "Framework", "Mode", "Model", "Run",
        "Acc", "N", "Input tok (effective)", "Output tok",
        "Cache read", "Cache create", "Cost",
    ]
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    # Sort: framework (agent before sdk), model, mode
    framework_order = {"agent": 0, "sdk": 1}
    mode_order = {"toc": 0, "raw": 1}
    rows_sorted = sorted(
        rows,
        key=lambda r: (framework_order.get(r["framework"], 9),
                       MODEL_LABELS.get(r["model"], r["model"]),
                       mode_order.get(r["mode"], 9),
                       r["run_name"]),
    )
    for r in rows_sorted:
        out.append("| " + " | ".join([
            r["framework"],
            r["mode"],
            MODEL_LABELS.get(r["model"], r["model"]),
            r["run_name"],
            _fmt_acc(r["accuracy"]),
            str(r["total_rows"]),
            _fmt_tokens(r["input_tokens"]),
            _fmt_tokens(r["output_tokens"]),
            _fmt_tokens(r["cache_read"]),
            _fmt_tokens(r["cache_create"]),
            _fmt_cost(r["cost_usd"]),
        ]) + " |")
    return "\n".join(out)


def _toc_vs_raw_deltas(rows: list[dict]) -> str:
    """Compute TOC vs Raw deltas within the same (framework, model)."""
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        key = (r["framework"], r["model"])
        grouped.setdefault(key, {})[r["mode"]] = r

    out = ["", "### TOC vs Raw deltas (within same framework + model)", ""]
    header = ["Framework", "Model",
              "Acc TOC", "Acc Raw", "ΔAcc",
              "Input TOC", "Input Raw", "ΔInput %",
              "Cost TOC", "Cost Raw"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for (fw, model), modes in sorted(grouped.items()):
        toc = modes.get("toc")
        raw = modes.get("raw")
        if not (toc and raw):
            continue
        it_toc = toc.get("input_tokens") or 0
        it_raw = raw.get("input_tokens") or 0
        delta_input_pct = (
            f"{(it_toc - it_raw) * 100 / it_raw:+.1f}%"
            if it_raw else "–"
        )
        a_toc = toc.get("accuracy")
        a_raw = raw.get("accuracy")
        delta_acc = (
            f"{(a_toc - a_raw) * 100:+.1f}pp"
            if a_toc is not None and a_raw is not None else "–"
        )
        out.append("| " + " | ".join([
            fw, MODEL_LABELS.get(model, model),
            _fmt_acc(a_toc), _fmt_acc(a_raw), delta_acc,
            _fmt_tokens(it_toc), _fmt_tokens(it_raw), delta_input_pct,
            _fmt_cost(toc.get("cost_usd")), _fmt_cost(raw.get("cost_usd")),
        ]) + " |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--md", type=Path, default=None, help="Also write table to this markdown file")
    p.add_argument("--only-sj", action="store_true", help="Restrict to dataset_S+J.jsonl")
    args = p.parse_args()

    rows = _collect()
    if args.only_sj:
        rows = [r for r in rows if r["dataset"] == "dataset_S+J.jsonl"]

    if not rows:
        print("No summary.json files found under result/agent or result/claude_sdk.")
        return

    table = _table_markdown(rows)
    deltas = _toc_vs_raw_deltas(rows)
    full = f"## Runs\n\n{table}\n{deltas}\n"
    print(full)

    if args.md:
        args.md.write_text(full, encoding="utf-8")
        print(f"\nWrote: {args.md}")


if __name__ == "__main__":
    main()
