import os
from .base_tool import BaseTool

MAX_CHARS = 20_000
PREVIEW_LINES = 80


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a file in the workspace. By default reads the whole file. "
        "For large files, the output is automatically truncated with head/tail preview. "
        "Use start_line and end_line (1-based) to read a specific range."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-based, inclusive). Optional.",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (1-based, inclusive). Optional.",
            },
        },
        "required": ["path"],
    }

    def run(self, path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
        try:
            safe_path = self.validate_path(path)

            with open(safe_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            total_lines = len(lines)

            if start_line is not None or end_line is not None:
                s = max(1, start_line or 1)
                e = min(total_lines, end_line or total_lines)
                selected = lines[s - 1 : e]
                content = "".join(selected)
                return {
                    "ok": True,
                    "total_lines": total_lines,
                    "showing": f"lines {s}-{e}",
                    "content": content,
                }

            full_content = "".join(lines)
            if len(full_content) <= MAX_CHARS:
                return {
                    "ok": True,
                    "total_lines": total_lines,
                    "content": full_content,
                }

            head = "".join(lines[:PREVIEW_LINES])
            tail = "".join(lines[-PREVIEW_LINES:])
            return {
                "ok": True,
                "total_lines": total_lines,
                "truncated": True,
                "head": head,
                "tail": tail,
                "message": (
                    f"File has {total_lines} lines and is too large to return in full. "
                    f"Showing first and last {PREVIEW_LINES} lines. "
                    f"Use start_line/end_line to read specific sections."
                ),
            }

        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }