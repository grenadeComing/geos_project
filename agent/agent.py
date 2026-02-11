import json
import logging
import os
from typing import Any, Dict, List, Tuple
from openai import OpenAI

from tools.base_tool import BaseTool
from tools.executation_tool import ExecutePHREEQCTool
from tools.read_file_tool import ReadFileTool
from tools.read_questions_tool import ReadQuestionsTool
from tools.write_file_tool import WriteFileTool
from tools.list_file_tool import ListFileTool

import time
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT = (BASE_DIR / "configs" / "system_prompt.txt").read_text(encoding="utf-8").strip()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
MAX_TOOL_ROUNDS = 6

TOOLS: Dict[str, Any] = {
    "read_file": ReadFileTool(),
    "read_questions": ReadQuestionsTool(),
    "write_file": WriteFileTool(),
    "execute_phreeqc": ExecutePHREEQCTool(),
    "list_file": ListFileTool(),
}


def _tool_spec(tool_obj: Any) -> Dict[str, Any]:
    params = getattr(tool_obj, "parameters", {})
    if isinstance(params, dict) and params.get("type"):
        spec_params = params
    else:
        spec_params = {
            "type": "object",
            "properties": {
                name: {"type": "string", "description": desc}
                for name, desc in params.items()
            },
            "required": list(params.keys()),
        }

    return {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": spec_params,
        },
    }


def _tool_call_payload(tool_call: Any) -> Dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments or "{}",
        },
    }


def _default_log_dir() -> Path:
    # assumes BaseTool.allowed_root points to workspace
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


def run_agent(
    messages: List[Dict[str, Any]],
    log_path: Path | None = None,
    max_steps: int = 24,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    One-tool-at-a-time agent loop:
    - If assistant returns no tools -> return its message.
    - If assistant returns tool call -> execute ONLY the first tool, append tool result, continue.
    """
    client = OpenAI()
    tool_specs = [_tool_spec(tool) for tool in TOOLS.values()]

    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + (messages or [])

    if log_path is None:
        log_path = _make_log_path()

    _log_event(log_path, "run_start", model=MODEL, max_steps=max_steps)

    # log the latest user message if present
    for m in reversed(messages):
        if m.get("role") == "user":
            _log_event(log_path, "user", content=m.get("content", ""))
            break

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tool_specs,
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message
        tool_calls = assistant_msg.tool_calls or []

        # Build assistant message payload (only first tool call)
        assistant_payload: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_msg.content or "",
        }
        first_tc = tool_calls[0] if tool_calls else None
        if first_tc is not None:
            assistant_payload["tool_calls"] = [_tool_call_payload(first_tc)]
        messages.append(assistant_payload)

        _log_event(
            log_path,
            "assistant",
            step=step,
            content=(assistant_msg.content or ""),
            tool_calls=(
                [
                    {
                        "id": first_tc.id,
                        "name": first_tc.function.name,
                        "arguments": first_tc.function.arguments,
                    }
                ]
                if first_tc is not None
                else []
            ),
        )

        # ✅ STOP CONDITION: no tools requested
        if first_tc is None:
            _log_event(log_path, "run_end", status="answered", step=step)
            return assistant_msg.content or "", messages

        # ✅ Execute ONLY ONE tool call
        tool_name = first_tc.function.name
        raw_args = first_tc.function.arguments or "{}"

        try:
            tool_args = json.loads(raw_args)
            if not isinstance(tool_args, dict):
                raise ValueError("Tool arguments must decode to a JSON object.")
        except Exception as e:
            result = {
                "ok": False,
                "error": f"Invalid tool arguments: {type(e).__name__}: {e}",
                "raw": raw_args,
            }
            args_for_log: Any = raw_args
        else:
            tool = TOOLS.get(tool_name)
            if tool is None:
                result = {"ok": False, "error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = tool.run(**tool_args)
                except Exception as e:
                    result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            args_for_log = tool_args

        messages.append(
            {
                "role": "tool",
                "tool_call_id": first_tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

        _log_event(
            log_path,
            "tool",
            step=step,
            tool=tool_name,
            args=args_for_log,
            result=result,
        )

        # continue loop: model sees tool result and decides next action

    _log_event(log_path, "run_end", status="max_steps_reached", step=max_steps)
    return "Reached max_steps without a final answer. Try rephrasing or increasing max_steps.", messages
