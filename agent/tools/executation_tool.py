import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from .base_tool import BaseTool


class ExecutePHREEQCTool(BaseTool):
    name = "execute_phreeqc"
    description = "Run PHREEQC on an input file (simple demo version)."
    parameters = {
        "input_path": "Path to the PHREEQC input file (e.g., inputs/example2.in.txt).",
    }

    # Hardcode for demo (your current paths)
    PHREEQC_BIN = "/Users/kezhang/Desktop/projects/geos/phreeqc-3.8.6-17100/src/phreeqc"
    DEFAULT_DB = "/Users/kezhang/Desktop/projects/geos/phreeqc-3.8.6-17100/database/phreeqc.dat"

    def run(self, input_path: str) -> Dict[str, Any]:
        try:
            if not os.path.isfile(self.PHREEQC_BIN):
                return {"ok": False, "error": f"PHREEQC binary not found: {self.PHREEQC_BIN}"}
            if not os.path.isfile(self.DEFAULT_DB):
                return {"ok": False, "error": f"PHREEQC database not found: {self.DEFAULT_DB}"}

            # Resolve relative paths within the currently allowed workspace.
            in_path = Path(self.validate_path(input_path))
            if not in_path.exists():
                return {"ok": False, "error": f"Input file not found: {in_path}"}

            # Run in the SAME folder as the input file so generated files are predictable
            workdir = in_path.parent
            out_file = workdir / "result.out"

            cmd = [self.PHREEQC_BIN, str(in_path), str(out_file), self.DEFAULT_DB]
            p = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)

            result_out = out_file.read_text(encoding="utf-8", errors="replace") if out_file.exists() else ""
            sel_file = workdir / "ex2.sel"
            sel_text = sel_file.read_text(encoding="utf-8", errors="replace") if sel_file.exists() else ""

            return {
                "ok": p.returncode == 0,
                "returncode": p.returncode,
                "cmd": cmd,
                "cwd": str(workdir),
                "stdout": p.stdout[-4000:],
                "stderr": p.stderr[-4000:],
                "result_out": result_out,
                "selected_output": sel_text,
            }

        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
