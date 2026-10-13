"""Prepare paired replay checkpoints from existing logs, without calling an LLM.

The checkpoint is immediately before delivery of the first successful PHREEQC
execution in a prespecified source run. Earlier writes and executions (including
failed executions) are replayed in order in a new directory. Final source
workspaces are never copied. Selection never uses predictions or value matches.

Example:
  python scripts/prepare_frozen_replay.py --limit 2 --output-root /tmp/replay-smoke

Output: manifest.json and item_XXXXX/{checkpoint.json,snapshot/}. The messages
omit the system message and stop with one pending execute_phreeqc tool call.
The continuation runner supplies a common mode-neutral system/tools and appends
payload_full or payload_toc as the reply to execute_tool_call_id. It must verify
the immutable snapshot and prevent further simulation or evidence-file writes;
branch answers and logs belong in separate directories.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "result/custom/rebuttal_mcq200_raw100k_rerun_gpt52_20260723"
DEFAULT_OUTPUT = REPO_ROOT / "result/custom/frozen_replay_gpt52/checkpoints"
DEFAULT_ITEMS = REPO_ROOT / "neurips_analysis/mechanism/auto/items.json"
DEFAULT_DATASET = REPO_ROOT / "datasets/neurips_dataset_158_verified.jsonl"
DEFAULT_BINARY = REPO_ROOT.parent / "phreeqc-3.8.6-17100/src/phreeqc"
DEFAULT_DATABASE = REPO_ROOT / "archive_acm_2026/all_materials/phreeqc.dat"
DEFAULT_REFERENCE = REPO_ROOT / "archive_acm_2026/all_materials/database_reference.txt"
SCHEMA_VERSION = 1


class Excluded(ValueError):
    """An auditable technical exclusion, unrelated to answer correctness."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_path(root: Path, relative: str) -> Path:
    """Resolve only relative, non-secret workspace paths; never follow escapes."""
    path = Path(str(relative))
    if path.is_absolute() or ".." in path.parts or "\\" in str(relative):
        raise Excluded("unsafe_workspace_path", str(relative))
    if any(part == ".env" or part.startswith(".env.") for part in path.parts):
        raise Excluded("secret_file_path", str(relative))
    resolved = (root / path).resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise Excluded("unsafe_workspace_path", str(relative))
    return resolved


def normalize_text(text: str, snapshot: Path | None = None, database: Path | None = None) -> str:
    """Use identical virtual paths in both arms; leave numerical evidence intact."""
    if snapshot is not None:
        text = text.replace(str(snapshot.resolve()), "/workspace")
    if database is not None:
        text = text.replace(str(database.resolve()), "/database/phreeqc.dat")
    text = re.sub(r'/[^\s\"\'<>]*?/work_space/dataset_q_\d{5}', "/workspace", text)
    text = re.sub(r'/[^\s\"\'<>]*?/archive_acm_2026/all_materials/phreeqc\.dat',
                  "/database/phreeqc.dat", text)
    return text


def normalize_value(value: Any, snapshot: Path | None = None, database: Path | None = None) -> Any:
    if isinstance(value, str):
        return normalize_text(value, snapshot, database)
    if isinstance(value, list):
        return [normalize_value(v, snapshot, database) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v, snapshot, database) for k, v in value.items()}
    return value


def read_events(log: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (ValueError, OSError) as error:
        raise Excluded("unreadable_log", str(error)) from error


def select_log(workspace: Path) -> tuple[Path, list[dict[str, Any]], int]:
    logs = sorted((workspace / "logs").glob("chat_*.jsonl"))
    if not logs:
        raise Excluded("missing_log")
    parsed = [(path, read_events(path)) for path in logs]
    complete = [(path, events) for path, events in parsed if any(e.get("action") == "run_end" for e in events)]
    path, events = (complete or parsed)[-1]
    return path, events, len(logs)


def first_success_index(events: list[dict[str, Any]]) -> int:
    for index, event in enumerate(events):
        if (event.get("action") == "tool" and event.get("tool") == "execute_phreeqc"
                and isinstance(event.get("result"), dict) and event["result"].get("ok") is True):
            return index
    raise Excluded("no_successful_execute")


def build_messages(events: list[dict[str, Any]], target_index: int) -> list[dict[str, Any]]:
    """Reconstruct visible history, with exactly the target tool result pending."""
    messages: list[dict[str, Any]] = []
    pending: dict[str, str] = {}
    for event in events[:target_index]:
        action = event.get("action")
        if action == "user":
            if pending:
                raise Excluded("incomplete_tool_history")
            messages.append({"role": "user", "content": event.get("content", "")})
        elif action == "assistant":
            if pending:
                raise Excluded("incomplete_tool_history")
            message: dict[str, Any] = {"role": "assistant", "content": event.get("content") or ""}
            calls = event.get("tool_calls") or []
            if not isinstance(calls, list):
                raise Excluded("unsupported_tool_call_encoding")
            formatted = []
            for call in calls:
                call_id, name = call.get("id"), call.get("name")
                if not call_id or not name or call_id in pending:
                    raise Excluded("invalid_tool_call")
                arguments = call.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                formatted.append({"id": call_id, "type": "function",
                                  "function": {"name": name, "arguments": arguments}})
                pending[call_id] = name
            if formatted:
                message["tool_calls"] = formatted
            messages.append(message)
        elif action == "tool":
            call_id = event.get("tool_call_id")
            if call_id not in pending or pending.pop(call_id) != event.get("tool"):
                raise Excluded("unmatched_tool_result", str(call_id))
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": json.dumps(normalize_value(event.get("result")), ensure_ascii=False)})
        elif action not in {"run_start", "aux_llm_usage"}:
            raise Excluded("unsupported_prefix_event", str(action))
    target = events[target_index]
    if pending != {target.get("tool_call_id"): "execute_phreeqc"}:
        raise Excluded("multiple_or_missing_pending_calls", str(pending))
    return normalize_value(messages)


def build_toc(path: Path, max_entries: int = 50) -> list[dict[str, Any]]:
    """Same section parser as evaluate_custom, without the historical size gate."""
    toc: list[dict[str, Any]] = []
    previous_dashes = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^-{3,}\s*([A-Za-z].+?)\s*-{3,}\s*$", stripped)
            if match:
                toc.append({"line": lineno, "section": match.group(1).strip()})
                previous_dashes = False
            elif re.match(r"^-{3,}$", stripped):
                previous_dashes = True
            elif previous_dashes:
                toc.append({"line": lineno, "section": stripped.rstrip(".")})
                previous_dashes = False
    if len(toc) > max_entries:
        half = max_entries // 2
        return toc[:half] + [{"line": -1, "section": f"... {len(toc) - max_entries} sections omitted ..."}] + toc[-half:]
    return toc


def validate_deck(snapshot: Path, input_file: Path, seen: set[Path] | None = None) -> None:
    """Reject PHREEQC paths that could read/write outside the new snapshot."""
    seen = seen if seen is not None else set()
    if input_file in seen:
        return
    seen.add(input_file)
    for line in input_file.read_text(encoding="utf-8").splitlines():
        code = line.split("#", 1)[0].strip()
        if not code:
            continue
        if re.match(r"DATABASE\b", code, re.I):
            raise Excluded("input_overrides_database", str(input_file.relative_to(snapshot)))
        if re.search(r"\b(?:SYSTEM|SHELL|OPEN|PLOT_XY)\b", code, re.I):
            raise Excluded("unsupported_input_external_io", code[:160])
        include = re.match(r"INCLUDE\$\s+(.+)", code, re.I)
        output = re.match(r"-?(?:file|dump)\s+(.+)", code, re.I)
        if include or output:
            name = (include or output).group(1).strip().strip('"\'')
            candidate = safe_path(input_file.parent, name)
            if snapshot.resolve() not in candidate.parents:
                raise Excluded("unsafe_input_file_path", name)
            if include:
                if not candidate.is_file():
                    raise Excluded("missing_include_file", name)
                validate_deck(snapshot, candidate, seen)


def comparable_report(text: str, snapshot: Path | None = None, database: Path | None = None) -> list[str]:
    normalized = normalize_text(text, snapshot, database)
    return [line for line in normalized.splitlines()
            if not re.match(r"\s*(?:Input file|Output file|Database file):", line)
            and "End of Run after" not in line and not re.fullmatch(r"\s*-+\s*", line)]


def replay_execution(snapshot: Path, event: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], Path, str]:
    tool_args = event.get("args") or {}
    input_file = safe_path(snapshot, tool_args.get("input_path", ""))
    if not input_file.is_file():
        raise Excluded("missing_reconstructed_input", str(tool_args.get("input_path")))
    validate_deck(snapshot, input_file)
    workdir = input_file.parent
    report = workdir / "result.out"
    if report.exists():
        report.unlink()
    before = set(workdir.iterdir())
    try:
        process = subprocess.run([str(args.phreeqc_bin), str(input_file), str(report), str(args.database)],
                                 cwd=workdir, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired as error:
        raise Excluded("reexecution_timeout", f"step {event.get('step')}, timeout {args.timeout}s") from error
    logged = event["result"]
    if (process.returncode == 0) != bool(logged.get("ok")):
        raise Excluded("reexecution_status_mismatch", f"step {event.get('step')}: recorded ok={logged.get('ok')}, replay rc={process.returncode}")
    if logged.get("returncode") is not None and process.returncode != logged["returncode"]:
        raise Excluded("reexecution_returncode_mismatch", f"step {event.get('step')}: {logged['returncode']} versus {process.returncode}")
    validation = "no_recorded_inline_payload"
    observed = logged.get("result_out_content")
    if isinstance(observed, str) and report.is_file():
        if "[TRUNCATED:" in observed:
            validation = "recorded_inline_truncated_not_compared"
        else:
            old = comparable_report(observed)
            new = comparable_report(report.read_text(encoding="utf-8", errors="replace"), snapshot, args.database)
            if old == new:
                validation = "exact_after_volatile_header_normalization"
            elif collections.Counter(old) == collections.Counter(new):
                validation = "same_lines_different_order"
            else:
                raise Excluded("reexecution_report_mismatch", f"step {event.get('step')}: recorded untruncated report differs")
    outputs = sorted(path for path in set(workdir.iterdir()) - before if path.is_file())
    result = {"ok": process.returncode == 0, "returncode": process.returncode,
              "stdout": normalize_text(process.stdout[-4000:], snapshot, args.database),
              "stderr": normalize_text(process.stderr[-4000:], snapshot, args.database),
              "output_files": [str(path.relative_to(snapshot)) for path in outputs],
              "file_sizes": {str(path.relative_to(snapshot)): path.stat().st_size for path in outputs}}
    return result, report, validation


def prepare_item(item: dict[str, Any], workspace: Path, item_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    log, events, n_logs = select_log(workspace)
    target_index = first_success_index(events)
    target = events[target_index]
    messages = build_messages(events, target_index)
    users = [e.get("content", "") for e in events[:target_index] if e.get("action") == "user"]
    if not users or item["question"].replace("\r", "").strip() not in users[0].replace("\r", ""):
        raise Excluded("question_workspace_mismatch")
    start = next((e for e in events if e.get("action") == "run_start"), {})
    step = target.get("step")
    max_steps = start.get("max_steps", args.source_max_steps)
    if not isinstance(step, int) or not isinstance(max_steps, int) or max_steps <= step:
        raise Excluded("no_remaining_steps_or_invalid_step", f"step={step}, max_steps={max_steps}")
    item_dir.mkdir()
    snapshot = item_dir / "snapshot"
    snapshot.mkdir()
    (snapshot / "logs").mkdir()
    shutil.copyfile(args.database_reference, snapshot / "database_reference.txt")
    executions = []
    prior_output_reads = 0
    metadata: dict[str, Any] = {}
    report = snapshot / "result.out"
    for event_index, event in enumerate(events[:target_index + 1]):
        if event.get("action") != "tool":
            continue
        name = event.get("tool")
        tool_args = event.get("args") or {}
        if not isinstance(tool_args, dict):
            raise Excluded("invalid_tool_arguments")
        if name == "write_file":
            if not isinstance(event.get("result"), dict) or event["result"].get("ok") is not True:
                raise Excluded("failed_write_in_prefix")
            path = safe_path(snapshot, tool_args.get("path", ""))
            if path.name == "final_answer.txt":
                raise Excluded("answer_written_before_checkpoint")
            content = tool_args.get("new_content")
            if not isinstance(content, str):
                raise Excluded("invalid_write_content")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        elif name == "execute_phreeqc":
            metadata, report, validation = replay_execution(snapshot, event, args)
            executions.append({"event_index": event_index, "step": event.get("step"),
                               "input_path": tool_args.get("input_path"),
                               "input_sha256": sha256_file(safe_path(snapshot, tool_args["input_path"])),
                               "recorded_ok": event["result"].get("ok"), "returncode": metadata["returncode"],
                               "report_validation": validation})
        elif name in {"read_file", "list_file", "grep_file"}:
            safe_path(snapshot, tool_args.get("path", "."))
            if name in {"read_file", "grep_file"} and Path(tool_args.get("path", "")).name != "database_reference.txt":
                prior_output_reads += 1
        else:
            raise Excluded("unsupported_prefix_tool", str(name))
    if not report.is_file():
        raise Excluded("successful_execution_without_report")
    # Path headers must not reveal a branch name; numerical output is unchanged.
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise Excluded("symlink_in_reconstructed_snapshot", str(path))
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="strict")
            normalized = normalize_text(text, snapshot, args.database)
            if normalized != text:
                path.write_text(normalized, encoding="utf-8")
    metadata["file_sizes"] = {name: safe_path(snapshot, name).stat().st_size for name in metadata["output_files"]}
    content = report.read_text(encoding="utf-8")
    full = dict(metadata)
    truncated = len(content) > args.max_full_chars
    if truncated:
        half = args.max_full_chars // 2
        full["result_out_content"] = (content[:half]
                                      + f"\n\n... [TRUNCATED: {len(content)} total chars, cap={args.max_full_chars}] ...\n\n"
                                      + content[-half:])
        full["warning"] = f"result.out truncated from {len(content)} to ~{args.max_full_chars} chars."
    else:
        full["result_out_content"] = content
    toc = dict(metadata)
    toc["result_out_toc"] = build_toc(report)
    toc["hint"] = "Use read_file with start_line/end_line to read specific sections."
    hashes = {str(path.relative_to(snapshot)): sha256_file(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}
    checkpoint = {"schema_version": SCHEMA_VERSION, "item_id": int(item["id"]), "item": item,
                  "model": start.get("model"), "messages": messages,
                  "execute_tool_call_id": target["tool_call_id"], "execute_step": step,
                  "remaining_steps": max_steps - step, "max_steps": max_steps, "source_max_steps": max_steps,
                  "snapshot": "snapshot", "snapshot_hashes": hashes,
                  "payload_full": full, "payload_toc": toc,
                  "report": {"path": str(report.relative_to(snapshot)), "sha256": sha256_file(report),
                             "chars": len(content), "bytes": report.stat().st_size,
                             "lines": len(content.splitlines()), "inline_truncated": truncated},
                  "source": {"run_dir": str(args.source_dir), "log_path": str(log), "log_sha256": sha256_file(log),
                             "log_count": n_logs, "selection": "latest_complete_else_latest",
                             "context_mode": start.get("context_mode"), "event_index": target_index,
                             "workspace_row": workspace.name, "prior_execute_count": len(executions) - 1,
                             "prior_nonreference_read_count": prior_output_reads},
                  "replayed_executions": executions,
                  "system_prompt_policy": "omitted; continuation runner supplies a common mode-neutral prompt",
                  "selection_policy": "first logged successful execute; no correctness or produced-value filter"}
    write_json(item_dir / "checkpoint.json", checkpoint)
    return checkpoint


def parse_item_ids(value: str) -> list[int]:
    try:
        return sorted({int(part) for part in value.split(",") if part.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integer item IDs") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--items-file", type=Path, default=DEFAULT_ITEMS, help="Only id/kind are read for selecting numeric items.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--phreeqc-bin", type=Path, default=Path(os.environ.get("PHREEQC_BIN", str(DEFAULT_BINARY))))
    parser.add_argument("--database", type=Path, default=Path(os.environ.get("PHREEQC_DB", str(DEFAULT_DATABASE))))
    parser.add_argument("--database-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--timeout", type=int, default=200)
    parser.add_argument("--max-full-chars", type=int, default=100000)
    parser.add_argument("--item-ids", "--ids", type=parse_item_ids)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    for name in ("source_dir", "output_root", "items_file", "dataset", "phreeqc_bin", "database", "database_reference"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.timeout < 1 or args.max_full_chars < 2 or (args.limit is not None and args.limit < 1):
        parser.error("timeout/limit must be positive and max-full-chars must be at least 2")
    if args.source_dir == args.output_root or args.source_dir in args.output_root.parents:
        parser.error("output-root must not be inside the source run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.items_file, args.dataset, args.phreeqc_bin, args.database, args.database_reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_root}. Use a new --output-root; existing outputs are never overwritten.")
    summary_path = args.source_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_dataset = REPO_ROOT / "datasets" / Path(summary["dataset"]).name
    source_rows = [json.loads(line) for line in source_dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    row_index = {int(row["id"]): index + 1 for index, row in enumerate(source_rows)}
    if len(row_index) != len(source_rows):
        raise ValueError("Source dataset has duplicate item IDs")
    verified = {int(row["id"]): row for row in (json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip())}
    item_meta = json.loads(args.items_file.read_text(encoding="utf-8"))
    numeric_ids = sorted(int(key) for key, meta in item_meta.items() if meta.get("kind") == "numeric" and int(key) in verified)
    selected = numeric_ids if args.item_ids is None else args.item_ids
    unknown = set(selected) - set(numeric_ids)
    if unknown:
        raise ValueError(f"Requested IDs are not verified numeric items: {sorted(unknown)}")
    if args.limit:
        selected = selected[:args.limit]
    args.source_max_steps = summary.get("max_steps", 24)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preparing", "source_dir": str(args.source_dir), "model": summary.get("model"),
        "all_numeric_item_ids": numeric_ids, "requested_item_ids": selected,
        "eligible": [], "excluded": [], "max_full_chars": args.max_full_chars,
        "toc_policy": "section TOC built at every report size; at most 50 entries plus omission marker",
        "report_validation_policy": "Compare untruncated logged reports after removing only path headers, elapsed-time lines, and pure dash rules; retain an explicit not-compared flag for truncated logged reports.",
        "selection_policy": "first logged successful execute in the prespecified source; no prediction/correctness/value-match filter",
        "files": {"source_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
                  "source_dataset": {"path": str(source_dataset), "sha256": sha256_file(source_dataset)},
                  "verified_dataset": {"path": str(args.dataset), "sha256": sha256_file(args.dataset)},
                  "numeric_items": {"path": str(args.items_file), "sha256": sha256_file(args.items_file)},
                  "phreeqc_binary": {"path": str(args.phreeqc_bin), "sha256": sha256_file(args.phreeqc_bin)},
                  "database": {"path": str(args.database), "sha256": sha256_file(args.database)},
                  "database_reference": {"path": str(args.database_reference), "sha256": sha256_file(args.database_reference)}},
        "limitations": ["Visible history is reconstructed; hidden provider reasoning state is unavailable.",
                        "Earlier failed executions and their visible results remain in the common prefix.",
                        "Checkpoints condition on a successful execution in one prespecified source protocol.",
                        "Inline reports exceeding max_full_chars are truncated; analyze the fits-inline subset separately.",
                        "No original log files or post-checkpoint files are copied into the snapshot."],
    }
    manifest_path = args.output_root / "manifest.json"
    write_json(manifest_path, manifest)
    for number, item_id in enumerate(selected, 1):
        item_dir = args.output_root / f"item_{item_id:05d}"
        try:
            if item_id not in row_index:
                raise Excluded("item_absent_from_source_dataset")
            workspace = args.source_dir / "work_space" / f"dataset_q_{row_index[item_id]:05d}"
            checkpoint = prepare_item(verified[item_id], workspace, item_dir, args)
            manifest["eligible"].append({"item_id": item_id, "checkpoint": str((item_dir / "checkpoint.json").relative_to(args.output_root)),
                                         "execute_step": checkpoint["execute_step"], "remaining_steps": checkpoint["remaining_steps"],
                                         "report_chars": checkpoint["report"]["chars"],
                                         "inline_truncated": checkpoint["report"]["inline_truncated"]})
            print(f"[{number}/{len(selected)}] item {item_id}: ready (step {checkpoint['execute_step']}, {checkpoint['report']['chars']} chars)", flush=True)
        except (Excluded, OSError, UnicodeError, ValueError, KeyError) as error:
            if item_dir.exists():
                shutil.rmtree(item_dir)
            manifest["excluded"].append({"item_id": item_id, "reason": getattr(error, "reason", "preparation_error"), "detail": str(error)})
            print(f"[{number}/{len(selected)}] item {item_id}: excluded ({getattr(error, 'reason', 'preparation_error')})", flush=True)
        write_json(manifest_path, manifest)
    manifest["status"] = "complete"
    manifest["counts"] = {"all_numeric": len(numeric_ids), "requested": len(selected),
                          "eligible": len(manifest["eligible"]), "excluded": len(manifest["excluded"])}
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["counts"], sort_keys=True), flush=True)
    return 0 if manifest["eligible"] else 1


if __name__ == "__main__":
    sys.exit(main())
