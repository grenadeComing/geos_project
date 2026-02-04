import os
from .base_tool import BaseTool


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the full content of a file within the allowed workspace."
    parameters = {
        "path": "Path to the file to read."
    }

    def run(self, path: str) -> dict:
        try:
            safe_path = self.validate_path(path)

            with open(safe_path, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "ok": True,
                "content": content
            }

        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}"
            }