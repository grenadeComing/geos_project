"""
Read hf_eval_results.jsonl and copy wrong/error/no-answer workspaces
into a review/ folder organized by failure category.

Drops a question.txt into each copied folder with the original question text
and correct answer so reviewers have full context at a glance.

Files larger than 10 MB are skipped to keep the review folder portable.

Usage:
    python collect_review.py
    python collect_review.py --results path/to/hf_eval_results.jsonl
"""
import argparse
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR / "work_space"
REVIEW_DIR = BASE_DIR / "review"
DEFAULT_RESULTS = BASE_DIR / "hf_eval_results.jsonl"
HF_CACHE_DIR = BASE_DIR / "hf_cache"
DATASET_ID = "grenadeComing/Phreeqc_MCQ"


def _load_questions(dataset_id: str) -> dict[int, dict]:
    """Load dataset and return a dict mapping 1-based index to row."""
    from datasets import load_dataset

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset_obj = load_dataset(dataset_id, cache_dir=str(HF_CACHE_DIR))
    if isinstance(dataset_obj, dict):
        dataset = dataset_obj[next(iter(dataset_obj.keys()))]
    else:
        dataset = dataset_obj

    lookup = {}
    for idx, row in enumerate(dataset):
        lookup[idx + 1] = {
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
        }
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect wrong/failed workspaces for review.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Path to hf_eval_results.jsonl")
    parser.add_argument("--output", type=Path, default=REVIEW_DIR, help="Review output directory")
    parser.add_argument("--dataset", default=DATASET_ID, help="HF dataset id")
    args = parser.parse_args()

    results = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]

    print("Loading dataset for question text...")
    questions = _load_questions(args.dataset)

    wrong = [r for r in results if r.get("is_correct") is False]
    errors = [r for r in results if r.get("error")]
    no_answer = [r for r in results if r.get("prediction") is None and not r.get("error")]

    to_review = {r["index"] for r in wrong + errors + no_answer}

    if not to_review:
        print("All questions correct — nothing to review.")
        return

    review_dir = args.output
    if review_dir.exists():
        shutil.rmtree(review_dir)

    wrong_dir = review_dir / "wrong"
    error_dir = review_dir / "errors"
    no_answer_dir = review_dir / "no_answer"

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    copied = 0
    skipped_files = 0
    for r in results:
        idx = r["index"]
        if idx not in to_review:
            continue
        src = WORKSPACE_ROOT / f"dataset_q_{idx:05d}"
        if not src.exists():
            continue

        if r.get("error"):
            dest = error_dir / src.name
        elif r.get("is_correct") is False:
            dest = wrong_dir / src.name
        else:
            dest = no_answer_dir / src.name

        dest.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if f.stat().st_size > MAX_FILE_SIZE:
                skipped_files += 1
                continue
            else:
                shutil.copy2(f, target)
        copied += 1

        q = questions.get(idx, {})
        question_txt = (
            f"Question #{idx}\n"
            f"{'=' * 60}\n\n"
            f"{q.get('question', '(not found)')}\n\n"
            f"{'=' * 60}\n"
            f"Correct answer: {q.get('answer', '?')}\n"
            f"Agent predicted: {r.get('prediction') or '(none)'}\n"
        )
        (dest / "question.txt").write_text(question_txt, encoding="utf-8")

    print(f"\nReview folder: {review_dir}")
    print(f"  Wrong answers:      {len(wrong)}")
    print(f"  Errors/crashes:     {len(errors)}")
    print(f"  No answer produced: {len(no_answer)}")
    print(f"  Copied folders:     {copied}")
    if skipped_files:
        print(f"  Large files skipped: {skipped_files} (>{MAX_FILE_SIZE // 1024 // 1024} MB)")


if __name__ == "__main__":
    main()
