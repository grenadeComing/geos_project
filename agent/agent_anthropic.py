"""
Anthropic Claude agent loop — same tools and system prompt as agent.py,
but uses the native Anthropic Messages API with tool_use.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anthropic

from tools.base_tool import BaseTool
from tools.executation_tool import ExecutePHREEQCTool
from tools.read_file_tool import ReadFileTool
from tools.write_file_tool import WriteFileTool
from tools.list_file_tool import ListFileTool

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT = (BASE_DIR / "configs" / "system_prompt.txt").read_text(encoding="utf-8").strip()

MODEL = "claude-sonnet-4-20250514"

TOOLS: Dict[str, Any] = {
    "read_file": ReadFileTool(),
    "write_file": WriteFileTool(),
    "execute_phreeqc": ExecutePHREEQCTool(),
    "list_file": ListFileTool(),
}


def _tool_spec(tool_obj: Any) -> Dict[str, Any]:
    """Convert a BaseTool into Anthropic's tool schema format."""
    params = getattr(tool_obj, "parameters", {})
    if isinstance(params, dict) and params.get("type"):
        input_schema = params
    else:
        input_schema = {
            "type": "object",
            "properties": {
                name: {"type": "string", "description": desc}
                for name, desc in params.items()
            },
            "required": list(params.keys()),
        }

    return {
        "name": tool_obj.name,
        "description": tool_obj.description,
        "input_schema": input_schema,
    }


def _default_log_dir() -> Path:
    root = Path(BaseTool.allowed_root).resolve() if BaseTool.allowed_root else BASE_DIR
    return root / "logs"


def _make_log_path() -> Path:
    log_dir = _default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"chat_{ts}_{int(time.time())}.jsonl"


def _log_event(log_path: Path, action: str, **data):
    event = {"ts": datetime.now().isoformat(timespec="seconds"), "action": action, **data}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


RESULT_ROOT = Path(__file__).resolve().parents[1] / "result"


def run_agent(
    messages: List[Dict[str, Any]],
    log_path: Path | None = None,
    max_steps: int = 24,
    model: str | None = None,
    provider: str = "anthropic",
) -> Tuple[str, List[Dict[str, Any]]]:
    if not BaseTool.allowed_root:
        raise RuntimeError("BaseTool.allowed_root must be set before calling run_agent.")
    resolved_root = Path(BaseTool.allowed_root).resolve()
    result_root = RESULT_ROOT.resolve()
    if result_root not in (resolved_root, *resolved_root.parents):
        raise RuntimeError(
            f"allowed_root ({resolved_root}) is not inside result ({result_root})."
        )

    use_model = model or MODEL
    client = anthropic.Anthropic(timeout=120.0)
    tool_specs = [_tool_spec(tool) for tool in TOOLS.values()]

    if log_path is None:
        log_path = _make_log_path()

    _log_event(log_path, "run_start", model=use_model, max_steps=max_steps)

    user_content = ""
    for m in messages:
        if m.get("role") == "user":
            user_content = m.get("content", "")
    _log_event(log_path, "user", content=user_content)

    anthropic_messages: list[dict] = [{"role": "user", "content": user_content}]

    for step in range(1, max_steps + 1):
        for attempt in range(1, 6):
            try:
                response = client.messages.create(
                    model=use_model,
                    system=SYSTEM_PROMPT,
                    messages=anthropic_messages,
                    tools=tool_specs,
                    max_tokens=4096,
                    temperature=0,
                )
                break
            except Exception as api_err:
                err_str = str(api_err)
                is_transient = any(k in err_str for k in ("429", "500", "502", "503", "529", "timeout", "connection", "overloaded"))
                if not is_transient or attempt == 5:
                    _log_event(log_path, "run_end", status="api_error", step=step, error=err_str)
                    raise
                logger.warning("API call failed (attempt %d/5, retrying): %s", attempt, api_err)
                time.sleep(2 ** attempt)

        text_parts = []
        tool_use_block = None
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use" and tool_use_block is None:
                tool_use_block = block

        assistant_text = "\n".join(text_parts)

        assistant_content: list[dict] = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use" and block.id == (tool_use_block.id if tool_use_block else None):
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        anthropic_messages.append({"role": "assistant", "content": assistant_content})

        _log_event(
            log_path,
            "assistant",
            step=step,
            content=assistant_text,
            tool_calls=(
                [{"id": tool_use_block.id, "name": tool_use_block.name, "arguments": json.dumps(tool_use_block.input)}]
                if tool_use_block else []
            ),
        )

        if tool_use_block is None or response.stop_reason == "end_turn":
            _log_event(log_path, "run_end", status="answered", step=step)
            return assistant_text, anthropic_messages

        tool_name = tool_use_block.name
        tool_args = tool_use_block.input or {}

        tool = TOOLS.get(tool_name)
        if tool is None:
            result = {"ok": False, "error": f"Unknown tool: {tool_name}"}
        else:
            try:
                if not isinstance(tool_args, dict):
                    raise ValueError("Tool arguments must be a dict.")
                result = tool.run(**tool_args)
            except Exception as e:
                result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

        anthropic_messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": json.dumps(result, ensure_ascii=False),
            }],
        })

        _log_event(
            log_path,
            "tool",
            step=step,
            tool=tool_name,
            args=tool_args,
            result=result,
        )

    _log_event(log_path, "run_end", status="max_steps_reached", step=max_steps)
    return "Reached max_steps without a final answer. Try rephrasing or increasing max_steps.", anthropic_messages
