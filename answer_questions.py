import sys
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent #repo folder
AGENT_DIR = BASE_DIR / "agent"
sys.path.insert(0, str(AGENT_DIR))

from tools.base_tool import BaseTool
from agent import run_agent


def build_prompt(file_name: str) -> str:
    return (
        "Answer the question in the file below.\n"
        "Return only the final answer.\n"
        "Format:\n"
        "1) <letter> <short answer>\n"
        "2) <letter> <short answer>\n"
        "\n"
        f"File: {file_name}\n"
        "Use read_questions to load the file content.\n"
        "Use only relative file paths for tool.\n"
    )


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

    results = []
    for path in question_files:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        question_workspace = workspace_root / _workspace_name_for(path)
        question_workspace.mkdir(parents=True, exist_ok=True)
        BaseTool.allowed_root = str(question_workspace)

        prompt = build_prompt(path.name)
        answer, _ = run_agent([{"role": "user", "content": prompt}])
        answer_text = answer.strip()

        (question_workspace / "final_answer.txt").write_text(
            answer_text + "\n", encoding="utf-8"
        )
        results.append(
            "\n".join(
                [
                    f"{path.name}",
                    f"workspace: {question_workspace.relative_to(BASE_DIR)}",
                    answer_text,
                ]
            )
        )

    output_path = workspace_root / "answers.txt"
    output_path.write_text("\n\n".join(results) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
