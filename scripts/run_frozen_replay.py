"""Paired read-only continuations of frozen GPT-5.2 PHREEQC checkpoints.

Preparation and --dry-run are offline. Only --run invokes the paid API.
Each pair shares history, files, prompt, tools and remaining assistant turns.
The sole between-arm change is the pending execute result: inline vs TOC.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import grade_row, parse_answer

REPO = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = """You are an expert geochemist and PHREEQC simulation specialist.
Answer the multiple-choice geochemistry question using the available PHREEQC output.
The simulation at the end of the conversation has completed. Its files are now frozen.
The pending execution result supplies metadata and either inline report content or a
table of contents with line numbers. Use the supplied content and tools as needed.

Available tools:
- list_file(path): List files/directories in the frozen workspace.
- read_file(path, start_line?, end_line?): Read a file, optionally by 1-based line range.
  Large files without a complete line range return their first and last 80 lines.
- write_file(path, new_content): Submit final_answer.txt only. No other writes are allowed.

Rules:
- Use relative file paths within the workspace.
- You cannot modify simulation files or execute another simulation in this stage.
- Base your answer on PHREEQC output, not memory. Do not guess.
- Call only one tool per turn.
- Submit your final answer with write_file("final_answer.txt", "<<< X >>>\\n"),
  where X is A, B, C, or D. Submission ends this continuation.
"""

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a frozen file, optionally by 1-based line range.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "start_line": {"type": "string"},
            "end_line": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_file", "description": "List files and directories in the frozen workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Submit final_answer.txt only; all other writes are forbidden.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "new_content": {"type": "string"}},
            "required": ["path", "new_content"]}}},
]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def build_messages(checkpoint: dict, condition: str) -> list[dict]:
    if condition not in {"full", "toc"}:
        raise ValueError("condition must be full or toc")
    messages = copy.deepcopy(checkpoint["messages"])
    if any(m["role"] == "system" for m in messages):
        raise ValueError("Checkpoint must not include a condition-specific system prompt")
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages + [{
        "role": "tool", "tool_call_id": checkpoint["execute_tool_call_id"],
        "content": json.dumps(checkpoint[f"payload_{condition}"], ensure_ascii=False),
    }]


def safe_path(root: Path, name: str) -> Path:
    if Path(name).is_absolute():
        raise ValueError("Use a relative path within the frozen workspace")
    root = root.resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Path is outside the frozen workspace")
    return target


def dispatch_readonly(snapshot: Path, answer_dir: Path, name: str, args: dict) -> dict:
    """No path supplied by a model can trigger execution or mutate a snapshot."""
    try:
        if name == "write_file":
            path = args.get("path", "")
            if path not in {"final_answer.txt", "./final_answer.txt"}:
                raise ValueError("Only final_answer.txt can be written; simulation files are frozen")
            content = args["new_content"]
            if not isinstance(content, str):
                raise ValueError("new_content must be a string")
            answer_dir.mkdir(parents=True, exist_ok=True)
            safe_path(answer_dir, "final_answer.txt").write_text(content, encoding="utf-8")
            return {"ok": True, "message": "Final answer submitted to final_answer.txt"}
        if name not in {"read_file", "list_file"}:
            raise ValueError(f"Tool {name!r} is unavailable; simulations cannot be rerun")
        target = safe_path(snapshot, args.get("path", "." if name == "list_file" else ""))
        if name == "list_file":
            return {"ok": True, "entries": sorted(p.name for p in target.iterdir())}
        content = target.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines(keepends=True)
        sl = int(args["start_line"]) if args.get("start_line") else None
        el = int(args["end_line"]) if args.get("end_line") else None
        if sl and el:
            return {"ok": True, "total_lines": len(lines), "showing": f"lines {sl}-{el}",
                    "content": "".join(lines[sl - 1:el])}
        if len(content) <= 20_000:
            return {"ok": True, "total_lines": len(lines), "content": content}
        return {"ok": True, "total_lines": len(lines),
                "showing": "first 80 + last 80 lines (file too large)",
                "content": "".join(lines[:80]) +
                f"\n\n... [{len(lines) - 160} lines omitted] ...\n\n" + "".join(lines[-80:])}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"ok": False, "error": str(exc)}


def snapshot_hashes(snapshot: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Snapshot contains a symlink: {path}")
        if path.is_file():
            hashes[str(path.relative_to(snapshot))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_checkpoint(path: Path) -> dict:
    cp = json.loads(path.read_text())
    if cp.get("schema_version") != 1:
        raise ValueError(f"Unsupported checkpoint schema: {path}")
    snapshot = (path.parent / cp.get("snapshot", "snapshot")).resolve()
    if not snapshot.is_dir() or not cp["snapshot_hashes"] or snapshot_hashes(snapshot) != cp["snapshot_hashes"]:
        raise ValueError(f"Frozen files no longer match checkpoint hashes: {path}")
    if type(cp["remaining_steps"]) is not int or cp["remaining_steps"] < 1:
        raise ValueError(f"No remaining API turns: {path}")
    if cp["remaining_steps"] != cp["max_steps"] - cp["execute_step"]:
        raise ValueError(f"Remaining turns do not match the historical checkpoint: {path}")
    report = safe_path(snapshot, cp["report"]["path"])
    if not report.is_file() or hashlib.sha256(report.read_bytes()).hexdigest() != cp["report"]["sha256"]:
        raise ValueError(f"Missing or changed report: {path}")
    payload_fields = {"result_out_content", "warning", "result_out_toc", "hint"}
    full, toc = cp["payload_full"], cp["payload_toc"]
    if ("result_out_content" not in full or "result_out_content" in toc or
            "result_out_toc" not in toc or "result_out_toc" in full or
            {k: v for k, v in full.items() if k not in payload_fields} !=
            {k: v for k, v in toc.items() if k not in payload_fields}):
        raise ValueError(f"Inline/TOC payload contract or shared metadata mismatch: {path}")
    pending, seen_ids = {}, set()
    for message in cp["messages"]:
        role = message.get("role")
        if role == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError(f"Unmatched tool response in checkpoint: {path}")
            del pending[message["tool_call_id"]]
        elif role in {"assistant", "user"}:
            if pending:
                raise ValueError(f"Missing historical tool response: {path}")
            for tc in message.get("tool_calls", []):
                if tc["id"] in seen_ids:
                    raise ValueError(f"Duplicate tool call ID: {path}")
                seen_ids.add(tc["id"])
                pending[tc["id"]] = tc["function"]["name"]
        else:
            raise ValueError(f"Unexpected historical role {role!r}: {path}")
    if pending != {cp["execute_tool_call_id"]: "execute_phreeqc"}:
        raise ValueError(f"Checkpoint does not end at the pending execution: {path}")
    # Build both before spending any money; this also checks the message contract.
    build_messages(cp, "full")
    build_messages(cp, "toc")
    return cp


class Budget:
    """Conservative standard-price estimate; a soft stop, not an invoice cap.

    Existing calls can finish after the threshold. Cached input discounts are not
    subtracted. Output usage already includes reasoning; do not add it again.
    """
    def __init__(self, limit_usd: float, spent_usd: float = 0.0):
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        self.stopped = False
        self.lock = threading.Lock()

    def available(self) -> bool:
        with self.lock:
            return not self.stopped and self.spent_usd < self.limit_usd

    def stop(self) -> None:
        with self.lock:
            self.stopped = True

    def add(self, usage: dict) -> float:
        cost = (usage.get("prompt_tokens", 0) * 1.75 +
                usage.get("completion_tokens", 0) * 14.0) / 1_000_000
        with self.lock:
            self.spent_usd += cost
        return cost


def response_dict(response: Any) -> dict:
    return response if isinstance(response, dict) else response.model_dump(mode="json")


def finish_result(cp: dict, state: dict, branch_dir: Path) -> dict:
    answer_path = branch_dir / "final_answer.txt"
    answer = answer_path.read_text() if answer_path.exists() else ""
    item = dict(cp["item"])
    item.setdefault("index", int(cp["item_id"]) - 1)
    item.setdefault("id", cp["item_id"])
    item.setdefault("task_type", "mcq")
    item.setdefault("gold", item.get("answer"))
    result = grade_row(item, parse_answer(answer, "mcq", strict=True))
    result.update(item_id=cp["item_id"], condition=state["condition"],
                  status=state["status"], calls=state["calls"],
                  estimated_cost_usd=state["estimated_cost_usd"],
                  input_tokens=sum(u.get("prompt_tokens", 0) for u in state["usage"]),
                  output_tokens=sum(u.get("completion_tokens", 0) for u in state["usage"]),
                  read_paths=state["read_paths"], checkpoint_fingerprint=state["fingerprint"],
                  remaining_steps=cp["remaining_steps"])
    atomic_json(branch_dir / "result.json", result)
    return result


def run_branch(checkpoint_path: Path, branch_dir: Path, condition: str,
               config: dict, budget: Budget, completion=None) -> dict:
    cp = load_checkpoint(checkpoint_path)
    snapshot = (checkpoint_path.parent / cp.get("snapshot", "snapshot")).resolve()
    fingerprint = digest({"checkpoint": cp, "condition": condition, "config": config,
                          "system": SYSTEM_PROMPT, "tools": TOOL_SPECS})
    branch_dir.mkdir(parents=True, exist_ok=True)
    state_path = branch_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["fingerprint"] != fingerprint:
            raise ValueError(f"Resume configuration changed: {branch_dir}")
        if state["phase"] == "done":
            return finish_result(cp, state, branch_dir)
        if state["phase"] == "request_pending":
            raise RuntimeError(f"Uncertain API call in {branch_dir}. Inspect state.json; "
                               "automatic replay is blocked to avoid paying twice.")
    else:
        state = {"fingerprint": fingerprint, "condition": condition, "phase": "ready",
                 "messages": build_messages(cp, condition), "calls": 0, "usage": [],
                 "estimated_cost_usd": 0.0, "read_paths": [], "events": []}
        atomic_json(state_path, state)
    if completion is None:
        from litellm import completion
    while True:
        if state["phase"] == "response_ready":
            response = state.pop("response")
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            assistant = {"role": "assistant", "content": message.get("content") or ""}
            if tool_calls:
                assistant["tool_calls"] = [{"id": tc["id"], "type": "function",
                                            "function": tc["function"]} for tc in tool_calls]
            state["messages"].append(assistant)
            submitted = False
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be an object")
                    result = dispatch_readonly(snapshot, branch_dir, name, args)
                except (ValueError, TypeError):
                    args, result = {}, {"ok": False, "error": "Invalid JSON tool arguments"}
                if name == "read_file":
                    state["read_paths"].append({"path": args.get("path"), "ok": result["ok"],
                                                "start_line": args.get("start_line"),
                                                "end_line": args.get("end_line")})
                state["messages"].append({"role": "tool", "tool_call_id": tc["id"],
                                          "content": json.dumps(result, ensure_ascii=False)})
                state["events"].append({"step": state["calls"], "tool": name,
                                        "args": args, "result": result})
                submitted |= name == "write_file" and result["ok"]
            if submitted:
                state.update(phase="done", status="submitted")
            elif not tool_calls:
                # Match the original file-based commitment rule; text alone is not a submission.
                state.update(phase="done", status="no_commit")
            else:
                state["phase"] = "ready"
            atomic_json(state_path, state)
        if state["phase"] == "done":
            return finish_result(cp, state, branch_dir)
        if state["calls"] >= cp["remaining_steps"]:
            state.update(phase="done", status="max_steps")
            atomic_json(state_path, state)
            return finish_result(cp, state, branch_dir)
        if not budget.available():
            return {"item_id": cp["item_id"], "condition": condition, "status": "budget_paused"}
        state["phase"] = "request_pending"
        atomic_json(state_path, state)
        # No automatic retries: an interrupted paid request must not be silently duplicated.
        try:
            response = response_dict(completion(
                model=config["model"], messages=state["messages"], tools=TOOL_SPECS,
                tool_choice="auto", parallel_tool_calls=False,
                max_completion_tokens=config["max_completion_tokens"],
                timeout=config["api_timeout"], num_retries=0,
            ))
        except Exception as exc:
            # Explicit request rejections are safe to retry on a later launch.
            # A timeout or server failure can follow a billed generation, so it
            # retains request_pending and requires inspection rather than retry.
            if getattr(exc, "status_code", None) in {400, 401, 403, 404, 422, 429}:
                state["phase"] = "ready"
                state["last_api_error"] = str(exc)
                atomic_json(state_path, state)
            budget.stop()
            raise
        usage = response.get("usage") or {}
        if not {"prompt_tokens", "completion_tokens"} <= usage.keys():
            budget.stop()
            raise RuntimeError("API response omitted token usage; stopped to avoid untracked spending")
        cost = budget.add(usage)
        state["calls"] += 1
        state["usage"].append(usage)
        state["estimated_cost_usd"] += cost
        state["response"] = response
        state["phase"] = "response_ready"
        atomic_json(state_path, state)


def collect_results(output: Path) -> dict:
    rows = []
    for path in sorted((output / "runs").glob("repeat_*/item_*/*/result.json")):
        row = json.loads(path.read_text())
        row["repeat"] = int(path.parents[2].name.removeprefix("repeat_"))
        rows.append(row)
    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    pairs = {}
    for row in rows:
        pairs.setdefault((row["item_id"], row["repeat"]), {})[row["condition"]] = row
    complete = [p for p in pairs.values() if set(p) == {"full", "toc"}]
    deltas_by_item = {}
    for (item_id, _), pair in pairs.items():
        if set(pair) == {"full", "toc"}:
            deltas_by_item.setdefault(item_id, []).append(
                int(pair["toc"]["is_correct"] is True) - int(pair["full"]["is_correct"] is True))
    summary = {"completed_branches": len(rows), "completed_pairs": len(complete),
               "incomplete_pairs": len(pairs) - len(complete),
               "estimated_cost_usd": sum(json.loads(p.read_text()).get("estimated_cost_usd", 0)
                   for p in (output / "runs").glob("repeat_*/item_*/*/state.json")),
               "conditions": {arm: {"n": len(rows_arm := [r for r in rows if r["condition"] == arm]),
                   "correct": sum(r["is_correct"] is True for r in rows_arm),
                   "no_commit": sum(r["is_correct"] is None for r in rows_arm)}
                   for arm in ("full", "toc")},
               "completed_items_with_a_pair": len(deltas_by_item),
               "completed_pair_item_mean_toc_minus_full_pp": (100 * sum(
                   sum(ds) / len(ds) for ds in deltas_by_item.values()) / len(deltas_by_item))
                   if deltas_by_item else None,
               "note": "Descriptive effect over completed pairs, averaged within then across items. "
                       "Interim coverage can be incomplete. Repeats are not independent items; "
                       "use item/setup-aware uncertainty for the manuscript."}
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "result/custom/frozen_replay_gpt52/checkpoints/manifest.json")
    parser.add_argument("--output", type=Path, default=REPO / "result/custom/frozen_replay_gpt52")
    parser.add_argument("--model", choices=["gpt-5.2"], default="gpt-5.2")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cost-usd", type=float, default=120,
                        help="Soft standard-price stop including prior usage; in-flight requests can finish.")
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--api-timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=20261011)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true", help="Start paid API calls; otherwise offline validation only.")
    action.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.repeats, args.workers, args.max_completion_tokens, args.max_cost_usd, args.api_timeout) <= 0:
        parser.error("Counts, timeout and budget must be positive")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Checkpoint preparation is incomplete or its manifest schema is unsupported")
    checkpoints = [(manifest_path.parent / entry["checkpoint"]).resolve() for entry in manifest["eligible"]]
    if not checkpoints:
        raise RuntimeError("No eligible checkpoints; inspect manifest exclusions")
    loaded = [load_checkpoint(path) for path in checkpoints]
    if len({cp["item_id"] for cp in loaded}) != len(loaded):
        raise ValueError("Manifest contains duplicate item checkpoints")
    if any(cp["item_id"] != entry["item_id"] for cp, entry in zip(loaded, manifest["eligible"])):
        raise ValueError("Manifest item ID does not match checkpoint")
    config = {"model": args.model, "max_completion_tokens": args.max_completion_tokens,
              "api_timeout": args.api_timeout,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    experiment = {"schema_version": 1, "config": config, "repeats": args.repeats,
                  "seed": args.seed, "system_prompt": SYSTEM_PROMPT, "tool_specs": TOOL_SPECS,
                  "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  "checkpoint_hashes": {str(cp["item_id"]): digest(cp) for cp in loaded}}
    print(f"Eligible items: {len(loaded)}; repeats: {args.repeats}; paired continuations: {len(loaded)*2*args.repeats}")
    print(f"Maximum remaining API calls: {sum(cp['remaining_steps'] for cp in loaded)*2*args.repeats}")
    print(f"Output: {args.output.resolve()}; estimated-cost soft stop: ${args.max_cost_usd:.2f}")
    if not args.run:
        print("OFFLINE CHECK PASSED. No API calls made. Add --run to execute.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; export it in your shell first")
    import litellm  # Verify the runtime before creating any branch.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Advisory lock prevents two launches from billing the same branches concurrently.
    import fcntl
    with (output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another replay is already using {output}") from None
        config_path = output / "experiment.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            # Increasing the target repeat count preserves all existing pairs:
            # the seeded schedule has the same prefix and branch fingerprints
            # exclude the repeat count. Never silently reduce the target.
            previous_repeats = previous.get("repeats", 0)
            if ({k: v for k, v in previous.items() if k != "repeats"} !=
                    {k: v for k, v in experiment.items() if k != "repeats"} or
                    args.repeats < previous_repeats):
                raise ValueError("Experiment settings changed or repeat count decreased. "
                                 "Use the existing settings or a different --output directory.")
        atomic_json(config_path, experiment)
        spent = sum(json.loads(p.read_text()).get("estimated_cost_usd", 0)
                    for p in (output / "runs").glob("repeat_*/item_*/*/state.json"))
        budget = Budget(args.max_cost_usd, spent)
        rng = random.Random(args.seed)
        jobs = []
        for repeat in range(1, args.repeats + 1):
            order = list(zip(checkpoints, loaded))
            rng.shuffle(order)
            for path, cp in order:
                arms = ["full", "toc"]
                rng.shuffle(arms)
                jobs.append((repeat, path, cp["item_id"], arms))

        def run_pair(job):
            repeat, path, item_id, arms = job
            results = []
            for arm in arms:
                branch = output / "runs" / f"repeat_{repeat:02d}" / f"item_{item_id:05d}" / arm
                results.append(run_branch(path, branch, arm, config, budget, litellm.completion))
            return results

        failures = []
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_pair, job): job for job in jobs}
                for count, future in enumerate(as_completed(futures), 1):
                    try:
                        results = future.result()
                        states = ", ".join(r["condition"] + ":" + r["status"] for r in results)
                        print(f"[{count}/{len(jobs)} pairs] {states}; estimated total ${budget.spent_usd:.2f}", flush=True)
                    except Exception as exc:
                        budget.stop()
                        job = futures[future]
                        failures.append({"repeat": job[0], "item_id": job[2], "error": str(exc)})
                        print(f"ERROR item {job[2]} repeat {job[0]}: {exc}", file=sys.stderr, flush=True)
        finally:
            summary = collect_results(output)
            atomic_json(output / "errors.json", failures)
            print(json.dumps(summary, indent=2), flush=True)
        if failures:
            raise SystemExit(1)
        if summary["completed_branches"] < len(jobs) * 2:
            print("Budget paused. Rerun the same command with a larger --max-cost-usd to continue.")
            raise SystemExit(2)


if __name__ == "__main__":
    main()
