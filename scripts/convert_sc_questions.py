"""
Build `datasets/phreeqc_bench_v2.jsonl` = 100 MCQ questions total:

  - 96 questions from Sachit's revised `mcqbuilder-data` repo
    (`mcqbuilder-data/phreeqc/sc/questions/question_{1..96}/question.json`)
  - 4 questions recovered from the old ACM-era dataset
    (`archive_acm_2026/dataset_Sachit+Jerry_for_ACM_paper.jsonl` rows 103, 108,
    113, 116 — the four Sachit deleted as near-duplicates of surviving
    Calcite questions).

We use the new 96 byte-for-byte (with Sachit's typo fixes, distractor
refresh, and the Q7 step 100→47 change applied), then append the 4 old
ones verbatim. Sequential ids 1..100.

v2 schema (see scripts/_common.py):
  {"id": int, "question": str, "task_type": "mcq", "gold": "A"|"B"|"C"|"D"}

Usage:
  python scripts/convert_sc_questions.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NEW_SRC_DIR = REPO_ROOT / "mcqbuilder-data" / "phreeqc" / "sc" / "questions"
OLD_SRC = REPO_ROOT / "archive_acm_2026" / "dataset_Sachit+Jerry_for_ACM_paper.jsonl"
DST = REPO_ROOT / "datasets" / "phreeqc_bench_v2.jsonl"

# The 4 rows Sachit deleted in the new version (1-indexed in the old file).
RECOVERED_OLD_ROWS = [103, 108, 113, 116]


def _check_letter(ans: str, src_label: str) -> str:
    a = str(ans).strip().upper()
    if a not in {"A", "B", "C", "D"}:
        raise ValueError(f"{src_label}: unexpected answer {ans!r}")
    return a


def load_new_96() -> list[dict]:
    """Load Sachit's 96 revised questions in numeric id order (question_1 .. question_96)."""
    if not NEW_SRC_DIR.is_dir():
        raise FileNotFoundError(f"Sachit's repo not found: {NEW_SRC_DIR}")

    rows: list[dict] = []
    for i in range(1, 97):
        fp = NEW_SRC_DIR / f"question_{i}" / "question.json"
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")
        raw = json.loads(fp.read_text(encoding="utf-8"))
        rows.append({
            "question": raw["question"],
            "gold": _check_letter(raw["answer"], f"question_{i}"),
            "_source": f"sc_new/question_{i}",
        })
    return rows


def load_recovered_4() -> list[dict]:
    """Load the 4 rows Sachit deleted, in the order given by RECOVERED_OLD_ROWS."""
    if not OLD_SRC.exists():
        raise FileNotFoundError(f"Old ACM dataset not found: {OLD_SRC}")

    wanted = set(RECOVERED_OLD_ROWS)
    by_row: dict[int, dict] = {}
    with OLD_SRC.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or lineno not in wanted:
                continue
            raw = json.loads(line)
            by_row[lineno] = {
                "question": raw["question"],
                "gold": _check_letter(raw["answer"], f"old_row_{lineno}"),
                "_source": f"acm_v1/row_{lineno}",
            }
    missing = wanted - by_row.keys()
    if missing:
        raise RuntimeError(f"Missing recovered rows: {sorted(missing)}")
    return [by_row[r] for r in RECOVERED_OLD_ROWS]


def main() -> None:
    new_rows = load_new_96()
    old_rows = load_recovered_4()
    all_rows = new_rows + old_rows

    if len(all_rows) != 100:
        raise RuntimeError(f"Expected 100 rows total, got {len(all_rows)}")

    # Assign sequential ids 1..100 and finalize schema.
    final: list[dict] = []
    for i, r in enumerate(all_rows, 1):
        final.append({
            "id": i,
            "question": r["question"],
            "task_type": "mcq",
            "gold": r["gold"],
        })

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = Counter(r["gold"] for r in final)
    print(f"Wrote {len(final)} rows to {DST}")
    print(f"  - {len(new_rows)} from Sachit's new repo (ids 1..{len(new_rows)})")
    print(f"  - {len(old_rows)} recovered from old ACM file "
          f"(ids {len(new_rows)+1}..{len(final)}): rows {RECOVERED_OLD_ROWS}")
    print(f"Answer distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
