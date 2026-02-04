import os
from .base_tool import BaseTool


class ListFileTool(BaseTool):
    name = "list_file"
    description = "List files and directories within the allowed workspace."
    parameters = {
        "path": "Directory path to list."
    }

    def run(self, path: str) -> dict:
        try:
            safe_path = self.validate_path(path)

            if not os.path.isdir(safe_path):
                return {
                    "ok": False,
                    "error": f"NotADirectoryError: {safe_path} is not a directory"
                }

            items = os.listdir(safe_path)

            return {
                "ok": True,
                "path": safe_path,
                "items": items
            }

        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}"
            }
