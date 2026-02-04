import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent #repo folder
AGENT_DIR = BASE_DIR / "agent"
sys.path.insert(0, str(AGENT_DIR))

from tools.base_tool import BaseTool
from agent import run_agent


def build_prompt(file_name: str) -> str:
    return (
        "Answer all questions in the file below.\n"
        "Return only the final answers.\n"
        "Format:\n"
        "1) <letter> <short answer>\n"
        "2) <letter> <short answer>\n"
        "\n"
        f"File: {file_name}\n"
        "Use read_questions to load the file content.\n"
    )


def main() -> None:
    agent_workspace = BASE_DIR / "agent_workspace"
    agent_workspace.mkdir(parents=True, exist_ok=True)
    BaseTool.allowed_root = str(agent_workspace)

    questions_dir = BASE_DIR / "questions"
    if not questions_dir.exists():
        raise FileNotFoundError(f"Questions folder not found: {questions_dir}")

    question_files = sorted([p for p in questions_dir.iterdir() if p.is_file()])
    if not question_files:
        raise FileNotFoundError(f"No question files found in: {questions_dir}")

    results = []
    for path in question_files:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        prompt = build_prompt(path.name)
        answer, _ = run_agent([{"role": "user", "content": prompt}])

        results.append(f"{path.name}\n{answer.strip()}")

    output_path = agent_workspace / "answers.txt"
    output_path.write_text("\n\n".join(results) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
