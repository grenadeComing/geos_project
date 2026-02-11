import sys
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent #repo folder
AGENT_DIR = BASE_DIR / "agent"
sys.path.insert(0, str(AGENT_DIR))

from tools.base_tool import BaseTool
from agent import run_agent


def build_prompt(file_name: str) -> str:
    return f"""Answer the multiple-choice question in the file below.
    File: {file_name}

    Steps:
    1) Use read_questions to read the question file.
    2) Solve it and determine X in {{A, B, C, D}}.
    3) Use write_file to create or overwrite final_answer.txt in the current workspace.
       The file content must be exactly: <<< X >>> followed by a newline.
"""

def _workspace_name_for(question_file: Path) -> str:
    stem = question_file.stem.strip()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return normalized or "question"


def _load_question_files(questions_dir: Path) -> list[Path]:
    split_files = sorted(questions_dir.glob("question_*.txt"))
    if split_files:
        return split_files
    return sorted([p for p in questions_dir.iterdir() if p.is_file()])


def main() -> None:
    workspace_root = BASE_DIR / "work_space"
    workspace_root.mkdir(parents=True, exist_ok=True)

    questions_dir = BASE_DIR / "questions"
    if not questions_dir.exists():
        raise FileNotFoundError(f"Questions folder not found: {questions_dir}")

    question_files = _load_question_files(questions_dir)
    if not question_files:
        raise FileNotFoundError(f"No question files found in: {questions_dir}")

    for path in question_files:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        question_workspace = workspace_root / _workspace_name_for(path)
        question_workspace.mkdir(parents=True, exist_ok=True)
        BaseTool.allowed_root = str(question_workspace)

        prompt = build_prompt(path.name)
        run_agent([{"role": "user", "content": prompt}])
        final_answer_path = question_workspace / "final_answer.txt"
        if not final_answer_path.exists():
            raise RuntimeError(
                f"Agent did not write final_answer.txt for {path.name} in {question_workspace}"
            )


if __name__ == "__main__":
    main()
