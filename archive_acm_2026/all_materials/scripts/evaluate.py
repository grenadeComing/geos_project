# python answer_questions.py --dataset dataset_sach.jsonl --provider openai --model gpt-5.1 --name sach_agent_gpt51_v2 --workers 4
import argparse
import json
import multiprocessing
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent #repo folder
AGENT_DIR = BASE_DIR / "agent"
sys.path.insert(0, str(AGENT_DIR))

from tools.base_tool import BaseTool
from agent import run_agent as run_agent_openai
from agent_anthropic import run_agent as run_agent_anthropic

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DATASET_FILE = BASE_DIR / "dataset_S+J.jsonl"
RESULT_ROOT = BASE_DIR / "result"


def _parse_choice(text: str | None) -> str | None:
    if not text:
        return None

    wrapped = re.search(r"<<<\s*([A-D])\s*>>>", text.strip(), flags=re.IGNORECASE)
    if wrapped:
        return wrapped.group(1).upper()
    return None


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


def _run_in_process(row: dict[str, Any], result_queue: multiprocessing.Queue, ws_root: str, model: str | None = None, provider: str = "openai", max_steps: int = 24) -> None:
    """Target function that runs inside an isolated process."""
    try:
        idx = row["index"]
        question_text = row["question"]

        workspace_name = f"dataset_q_{idx + 1:05d}"
        question_workspace = Path(ws_root) / workspace_name
        question_workspace.mkdir(parents=True, exist_ok=True)
        BaseTool.allowed_root = str(question_workspace)

        db_ref = BASE_DIR / "database_reference.txt"
        if db_ref.exists():
            shutil.copy(db_ref, question_workspace / "database_reference.txt")

        prompt = build_prompt(question_text)
        if provider == "anthropic":
            run_agent_anthropic([{"role": "user", "content": prompt}], model=model, max_steps=max_steps)
        else:
            run_agent_openai([{"role": "user", "content": prompt}], model=model, provider=provider)
    except Exception as e:
        result_queue.put({"error": str(e)})
        return

    result_queue.put({"ok": True})


def _process_one_question(row: dict[str, Any], ws_root: str, model: str | None = None, provider: str = "openai", max_steps: int = 24) -> dict[str, Any]:
    """Spawn an isolated process for one question. If it crashes, only this question fails."""
    idx = row["index"]
    truth_raw = row["answer"]
    truth = str(truth_raw).strip().upper()
    if truth not in {"A", "B", "C", "D"}:
        truth = None

    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run_in_process, args=(row, result_queue, ws_root, model, provider, max_steps))
    proc.start()
    timeout = 600 if provider == "anthropic" else 300
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {
            "index": idx + 1,
            "truth": truth,
            "prediction": None,
            "is_correct": None,
            "error": "Timed out after 300 seconds",
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


def _load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset file not found: {dataset_path}")

    records: list[dict[str, Any]] = []
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
            records.append(
                {
                    "index": idx,
                    "question": str(question),
                    "answer": str(answer),
                }
            )

    if not records:
        raise RuntimeError(f"No usable rows found in {dataset_path}.")
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PHREEQC agent over Hugging Face MCQ dataset and evaluate answers."
    )
    parser.add_argument("--dataset", default=str(DATASET_FILE), help="Path to dataset JSONL file")
    parser.add_argument("--provider", default="openai", choices=["openai", "google", "anthropic"], help="API provider")
    parser.add_argument("--model", default=None, help="Model name (default: gpt-5.2)")
    parser.add_argument("--name", default=None, help="Run name for the result subfolder (default: auto timestamp)")
    parser.add_argument("--resume", action="store_true", help="Skip questions that already have final_answer.txt")
    parser.add_argument("--max-steps", type=int, default=24, help="Max agent steps per question (default: 24)")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel workers (default: 6). Use >1 for parallel processing.",
    )
    return parser.parse_args()


def main() -> None:
    global WORKSPACE_ROOT

    args = _parse_args()
    records = _load_dataset(Path(args.dataset))

    run_name = args.name or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RESULT_ROOT / "agent" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    WORKSPACE_ROOT = run_dir / "work_space"
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    workers = max(1, args.workers)
    results_list: list[dict[str, Any]] = []
    ws_root = str(WORKSPACE_ROOT)

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
            fa_path = WORKSPACE_ROOT / f"dataset_q_{idx + 1:05d}" / "final_answer.txt"
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

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one_question, row, ws_root, args.model, args.provider, args.max_steps): row for row in records_to_run}
        iterator = (
            tqdm(as_completed(futures), total=len(futures), desc="Evaluating", unit="q")
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
        "provider": args.provider,
        "model": args.model or {"openai": "gpt-5.2", "google": "gemini-2.5-pro", "anthropic": "claude-sonnet-4-20250514"}.get(args.provider, "gpt-5.2"),
        "method": "agent",
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
