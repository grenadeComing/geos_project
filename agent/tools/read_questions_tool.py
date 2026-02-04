from pathlib import Path
from typing import Dict, Any

from .base_tool import BaseTool


class ReadQuestionsTool(BaseTool):
    name = "read_questions"
    description = "Read a question file from the repository questions folder (read-only)."
    parameters = {
        "file_name": "Question file name under the questions/ directory."
    }

    def run(self, file_name: str) -> Dict[str, Any]:
        try:
            # Resolve repository root from this file's location.
            repo_root = Path(__file__).resolve().parents[2]
            questions_dir = repo_root / "questions"
            target = (questions_dir / file_name).resolve()

            # Enforce read-only access to the questions directory only.
            if questions_dir not in target.parents:
                return {"ok": False, "error": f"Disallowed path: {target}"}
            if not target.exists():
                return {"ok": False, "error": f"Question file not found: {target}"}

            content = target.read_text(encoding="utf-8")
            return {"ok": True, "content": content}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
