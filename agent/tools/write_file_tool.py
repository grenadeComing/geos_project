import os
from pathlib import Path
from .base_tool import BaseTool

class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Create a new text file or fully overwrite an existing one with the provided content. "
    )
    parameters = {
        "path": "Target file path",
        "new_content": "Full file content"
    }

    def run(self, path: str, new_content: str):
        safe_base_path = self.validate_path(path)
        os.makedirs(os.path.dirname(safe_base_path), exist_ok=True)

        try:
            Path(safe_base_path).write_text(new_content, encoding="utf-8")
            return {
                "ok": True,
                "message": f"File content successfully written to: {safe_base_path}"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to write file: {repr(e)}"
            }