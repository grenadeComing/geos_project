import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from .base_tool import BaseTool

class ExecutePHREEQCTool(BaseTool):
    name = "execute_phreeqc"
    description = (
        "Run PHREEQC on an input file. Returns execution status, paths to "
        "output files, and a table-of-contents with section line numbers for "
        "result.out. Use read_file with start_line/end_line to inspect specific sections."
    )
    parameters = {
        "input_path": "Path to the PHREEQC input file (e.g., inputs/example2.in.txt).",
    }

    PHREEQC_BIN = "/Users/kezhang/Desktop/projects/geos/phreeqc-3.8.6-17100/src/phreeqc"
    DEFAULT_DB = "/Users/kezhang/Desktop/projects/geos/phreeqc-3.8.6-17100/database/phreeqc.dat"

    def _collect_output_files(self, workdir: Path, before: set[str]) -> List[str]:
        """Return workspace-relative paths for files created or modified by the run."""
        root = Path(self.allowed_root)
        after = set(os.listdir(workdir))
        new_files = sorted(after - before)
        result = []
        for name in new_files:
            full = workdir / name
            if full.is_file():
                try:
                    result.append(str(full.relative_to(root)))
                except ValueError:
                    result.append(str(full))
        return result

    @staticmethod
    def _build_toc(filepath: Path) -> List[Dict[str, Any]]:
        """Scan result.out for section delimiters and return a table of contents."""
        toc: List[Dict[str, Any]] = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                prev_dashes_line = 0
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    is_dashes = bool(re.match(r"^-{3,}$", stripped))
                    # "----Saturation indices----" — label embedded between dashes
                    m = re.match(r"^-{3,}\s*([A-Za-z].+?)\s*-{3,}\s*$", stripped)
                    if m:
                        toc.append({"line": lineno, "section": m.group(1).strip()})
                        prev_dashes_line = 0
                        continue
                    if is_dashes:
                        prev_dashes_line = lineno
                        continue
                    if prev_dashes_line:
                        toc.append({"line": lineno, "section": stripped.rstrip(".")})
                        prev_dashes_line = 0
                    else:
                        prev_dashes_line = 0
        except Exception:
            pass
        return toc

    def run(self, input_path: str) -> Dict[str, Any]:
        try:
            if not os.path.isfile(self.PHREEQC_BIN):
                return {"ok": False, "error": f"PHREEQC binary not found: {self.PHREEQC_BIN}"}
            if not os.path.isfile(self.DEFAULT_DB):
                return {"ok": False, "error": f"PHREEQC database not found: {self.DEFAULT_DB}"}

            in_path = Path(self.validate_path(input_path))
            if not in_path.exists():
                return {"ok": False, "error": f"Input file not found: {in_path}"}

            workdir = in_path.parent
            out_file = workdir / "result.out"
            files_before = set(os.listdir(workdir))

            cmd = [self.PHREEQC_BIN, str(in_path), str(out_file), self.DEFAULT_DB]
            p = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)

            output_files = self._collect_output_files(workdir, files_before)

            toc = self._build_toc(out_file) if out_file.exists() else []

            return {
                "ok": p.returncode == 0,
                "returncode": p.returncode,
                "stdout": p.stdout[-4000:],
                "stderr": p.stderr[-4000:],
                "output_files": output_files,
                "result_out_toc": toc,
                "hint": "Use read_file with start_line/end_line to read specific sections from the TOC above.",
            }

        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
