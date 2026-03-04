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
    DEFAULT_DB = "/Users/kezhang/Desktop/projects/geos/geos_github/phreeqc.dat"

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

    MAX_TOC_ENTRIES = 50

    @staticmethod
    def _build_toc(filepath: Path) -> List[Dict[str, Any]]:
        """Scan result.out for section delimiters and return a table of contents.
        If there are too many sections (e.g. transport simulations with thousands
        of time steps), keep only the first and last entries so the agent can
        find the setup and final results."""
        toc: List[Dict[str, Any]] = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                prev_dashes_line = 0
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    is_dashes = bool(re.match(r"^-{3,}$", stripped))
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

        limit = ExecutePHREEQCTool.MAX_TOC_ENTRIES
        if len(toc) > limit:
            half = limit // 2
            trimmed = toc[:half] + [{"line": -1, "section": f"... {len(toc) - limit} sections omitted ..."}] + toc[-half:]
            return trimmed
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

            MAX_TOC_SIZE = 50 * 1024 * 1024  # 50 MB
            toc = []
            toc_skipped = False
            if out_file.exists():
                if out_file.stat().st_size <= MAX_TOC_SIZE:
                    toc = self._build_toc(out_file)
                else:
                    toc_skipped = True

            file_sizes = {}
            for f in output_files:
                full = Path(self.allowed_root) / f
                if full.exists():
                    file_sizes[f] = full.stat().st_size

            result = {
                "ok": p.returncode == 0,
                "returncode": p.returncode,
                "stdout": p.stdout[-4000:],
                "stderr": p.stderr[-4000:],
                "output_files": output_files,
                "file_sizes": file_sizes,
                "result_out_toc": toc,
                "hint": "Use read_file with start_line/end_line to read specific sections.",
            }
            if toc_skipped:
                result["warning"] = (
                    f"result.out is very large ({out_file.stat().st_size / 1e6:.0f} MB). "
                    "TOC scan skipped. Use read_file to inspect the tail of the file for final results."
                )
            return result

        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
