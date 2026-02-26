import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent #repo folder
AGENT_DIR = BASE_DIR / "agent"
sys.path.insert(0, str(AGENT_DIR))

from tools.base_tool import BaseTool
from agent import run_agent

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DATASET_ID = "grenadeComing/Phreeqc_MCQ"
RESULTS_FILE = "hf_eval_results.jsonl"
SUMMARY_FILE = "hf_eval_summary.json"
HF_CACHE_DIR = BASE_DIR / "hf_cache"
WORKSPACE_ROOT = BASE_DIR / "work_space"


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


def _process_one_question(row: dict[str, Any]) -> dict[str, Any]:
    """Process a single question in a worker process. Returns a result dict."""
    idx = row["index"]
    question_text = row["question"]
    truth_raw = row["answer"]
    truth = str(truth_raw).strip().upper()
    if truth not in {"A", "B", "C", "D"}:
        truth = None

    workspace_name = f"dataset_q_{idx + 1:05d}"
    question_workspace = WORKSPACE_ROOT / workspace_name
    question_workspace.mkdir(parents=True, exist_ok=True)
    BaseTool.allowed_root = str(question_workspace)

    prompt = build_prompt(question_text)
    run_agent([{"role": "user", "content": prompt}])

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


def _load_dataset(dataset_id: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install it with: pip install datasets"
        ) from exc

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset_obj = load_dataset(dataset_id, cache_dir=str(HF_CACHE_DIR))
    if isinstance(dataset_obj, dict):
        if not dataset_obj:
            raise RuntimeError(f"No splits found in dataset '{dataset_id}'.")
        first_split = next(iter(dataset_obj.keys()))
        dataset = dataset_obj[first_split]
    else:
        dataset = dataset_obj
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(dataset):
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
        raise RuntimeError(
            f"No usable rows found in dataset '{dataset_id}'."
        )
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PHREEQC agent over Hugging Face MCQ dataset and evaluate answers."
    )
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset id")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel workers (default: 1). Use >1 for parallel processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    records = _load_dataset(args.dataset)

    results_path = BASE_DIR / RESULTS_FILE
    summary_path = BASE_DIR / SUMMARY_FILE

    workers = max(1, args.workers)
    results_list: list[dict[str, Any]] = []

    with results_path.open("w", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one_question, row): row for row in records}
            iterator = (
                tqdm(as_completed(futures), total=len(futures), desc="Evaluating", unit="q")
                if tqdm is not None
                else as_completed(futures)
            )
            for future in iterator:
                try:
                    result_row = future.result()
                    results_list.append(result_row)
                except Exception as e:
                    row = futures[future]
                    results_list.append({
                        "index": row["index"] + 1,
                        "truth": str(row.get("answer", "")).strip().upper() or None,
                        "prediction": None,
                        "is_correct": None,
                        "error": str(e),
                    })

        # Write results in index order for stable output
        results_list.sort(key=lambda r: r["index"])
        for result_row in results_list:
            out.write(json.dumps(result_row, ensure_ascii=False) + "\n")

    total = len(results_list)
    parsed = sum(1 for r in results_list if r.get("prediction") is not None and r.get("truth") is not None)
    correct = sum(1 for r in results_list if r.get("is_correct") is True)
    accuracy = (correct / parsed) if parsed else 0.0
    summary = {
        "dataset": args.dataset,
        "workers": workers,
        "total_rows": total,
        "parsed_rows": parsed,
        "correct_rows": correct,
        "accuracy": accuracy,
        "results_file": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
