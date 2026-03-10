"""
Baseline evaluation: one-shot OpenAI models, no agent / no tools.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "dataset_S+J.jsonl"
RESULT_ROOT = BASE_DIR / "result"
MODEL = "gpt-5.2"

SYSTEM_PROMPT = (
    "You are a geochemistry expert. "
    "Answer the multiple-choice question. "
    "Reply with ONLY your answer in the form <<< X >>> where X is A, B, C, or D. "
    "Do not explain."
)


def _parse_choice(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"<<<\s*([A-D])\s*>>>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1).upper()
    return None


def _query_model(client: OpenAI, question: str, model: str) -> str | None:
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_completion_tokens=32,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            transient = any(k in err for k in ("429", "500", "502", "503", "timeout", "connection"))
            if not transient or attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return None


def _process_one(row: dict[str, Any], client: OpenAI, model: str) -> dict[str, Any]:
    idx = row["index"]
    truth_raw = row["answer"]
    truth = str(truth_raw).strip().upper()
    if truth not in {"A", "B", "C", "D"}:
        truth = None

    try:
        raw = _query_model(client, row["question"], model)
        pred = _parse_choice(raw)
    except Exception as e:
        return {
            "index": idx + 1,
            "truth": truth,
            "prediction": None,
            "is_correct": None,
            "error": str(e),
        }

    return {
        "index": idx + 1,
        "truth": truth,
        "prediction": pred,
        "is_correct": (pred == truth) if pred and truth else None,
        "raw_response": raw,
    }


def _load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset file not found: {dataset_path}")

    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("question")
            a = row.get("answer")
            if not q or a is None:
                continue
            records.append({"index": idx, "question": str(q), "answer": str(a)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline: one-shot OpenAI, no tools.")
    parser.add_argument("--dataset", default=str(DATASET_FILE), help="Path to dataset JSONL file")
    parser.add_argument("--model", default=MODEL, help="Model name (default: gpt-5.2)")
    parser.add_argument("--name", default=None, help="Run name for the result subfolder")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    model = args.model
    client = OpenAI()
    records = _load_dataset(Path(args.dataset))

    run_name = args.name or f"baseline_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RESULT_ROOT / "oneshot" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    results_list: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_process_one, row, client, model): row
            for row in records
        }
        iterator = (
            tqdm(as_completed(futures), total=len(futures), desc=f"Baseline ({model})", unit="q")
            if tqdm
            else as_completed(futures)
        )
        for future in iterator:
            results_list.append(future.result())

    results_list.sort(key=lambda r: r["index"])

    with results_path.open("w", encoding="utf-8") as f:
        for r in results_list:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results_list)
    correct = sum(1 for r in results_list if r.get("is_correct") is True)
    errors = sum(1 for r in results_list if r.get("error"))
    accuracy = (correct / total) if total else 0.0

    summary = {
        "provider": "openai",
        "model": model,
        "method": "one_shot_no_tools",
        "dataset": str(Path(args.dataset).name),
        "run_name": run_name,
        "total_rows": total,
        "correct_rows": correct,
        "error_rows": errors,
        "accuracy": accuracy,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results saved to: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
