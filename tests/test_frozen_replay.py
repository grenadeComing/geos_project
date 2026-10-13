"""Offline checks for frozen-computation replay. These tests never call an API."""
from __future__ import annotations

import importlib.util
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_frozen_replay", REPO / "scripts" / "run_frozen_replay.py"
)
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


class WorkspaceFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.answers = self.root / "answers"
        self.answers.mkdir()
        (self.snapshot / "result.out").write_text("report line one\nreport line two\n")
        (self.snapshot / "input.pqi").write_text("SOLUTION 1\nEND\n")
        self.original = self.snapshot_bytes()

    def snapshot_bytes(self):
        return {
            str(p.relative_to(self.snapshot)): p.read_bytes()
            for p in self.snapshot.rglob("*")
            if p.is_file() and not p.is_symlink()
        }

    def call(self, name, **arguments):
        return replay.dispatch_readonly(self.snapshot, self.answers, name, arguments)


class ReadOnlyWorkspaceTests(WorkspaceFixture, unittest.TestCase):
    def test_simulation_and_unknown_tools_cannot_execute(self):
        for name in ("execute_phreeqc", "run_command", "delete_file"):
            with self.subTest(tool=name):
                result = self.call(name, input_path="input.pqi", command="touch forbidden")
                self.assertFalse(result.get("ok", False), result)
        self.assertEqual(self.original, self.snapshot_bytes())
        self.assertEqual([], list(self.answers.iterdir()))

    def test_only_final_answer_can_be_written_and_snapshot_is_unchanged(self):
        for path in ("input.pqi", "result.out", "selected.out", "other.txt"):
            with self.subTest(path=path):
                result = self.call("write_file", path=path, new_content="changed")
                self.assertFalse(result.get("ok", False), result)
        result = self.call("write_file", path="final_answer.txt", new_content="<<< B >>>")
        self.assertTrue(result.get("ok", False), result)
        self.assertEqual("<<< B >>>", (self.answers / "final_answer.txt").read_text())
        self.assertEqual(self.original, self.snapshot_bytes())

    def test_parent_and_absolute_paths_cannot_escape_the_snapshot(self):
        outside = self.root / "outside.txt"
        outside.write_text("PRIVATE_OUTSIDE_MARKER")
        for path in ("../outside.txt", str(outside)):
            with self.subTest(path=path):
                result = self.call("read_file", path=path)
                self.assertFalse(result.get("ok", False), result)
                self.assertNotIn("PRIVATE_OUTSIDE_MARKER", json.dumps(result))
        result = self.call("write_file", path="../outside.txt", new_content="changed")
        self.assertFalse(result.get("ok", False), result)
        self.assertEqual("PRIVATE_OUTSIDE_MARKER", outside.read_text())

    def test_symlink_cannot_read_outside_or_redirect_final_answer(self):
        outside = self.root / "outside.txt"
        outside.write_text("PRIVATE_OUTSIDE_MARKER")
        (self.snapshot / "escape.txt").symlink_to(outside)
        result = self.call("read_file", path="escape.txt")
        self.assertFalse(result.get("ok", False), result)
        self.assertNotIn("PRIVATE_OUTSIDE_MARKER", json.dumps(result))
        (self.answers / "final_answer.txt").symlink_to(outside)
        result = self.call("write_file", path="final_answer.txt", new_content="changed")
        self.assertFalse(result.get("ok", False), result)
        self.assertEqual("PRIVATE_OUTSIDE_MARKER", outside.read_text())
        self.assertEqual(self.original, self.snapshot_bytes())

    def test_large_report_has_a_preview_and_explicit_ranges_are_available(self):
        lines = [f"LINE-{n:03d} " + "x" * 100 + "\n" for n in range(1, 401)]
        (self.snapshot / "large.out").write_text("".join(lines))
        result = self.call("read_file", path="large.out")
        self.assertTrue(result["ok"])
        self.assertIn("LINE-001", result["content"])
        self.assertIn("LINE-400", result["content"])
        self.assertNotIn("LINE-200", result["content"])
        selected = self.call("read_file", path="large.out", start_line="199", end_line="201")
        self.assertTrue(selected["ok"])
        self.assertEqual("".join(lines[198:201]), selected["content"])
        self.assertEqual("".join(lines), (self.snapshot / "large.out").read_text())


def tool_response(name="write_file", arguments=None, *, input_tokens=100, output_tokens=10):
    if arguments is None:
        arguments = {"path": "final_answer.txt", "new_content": "<<< B >>>\n"}
    return {
        "choices": [{"message": {
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "fake-call", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        }}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
    }


class ReplayStateTests(WorkspaceFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config = {"model": "gpt-5.2", "max_completion_tokens": 4096, "api_timeout": 120}
        self.checkpoint_path = self.root / "checkpoint.json"
        self.checkpoint = {
            "schema_version": 1,
            "item_id": 2,
            "item": {"id": 2, "question": "What is the pH?", "answer": "B",
                     "private_metadata": "DO_NOT_SEND_SCORING_METADATA"},
            "messages": [
                {"role": "user", "content": "What is the pH?"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "execute-1", "type": "function",
                    "function": {"name": "execute_phreeqc",
                                 "arguments": '{"input_path":"input.pqi"}'},
                }]},
            ],
            "execute_tool_call_id": "execute-1",
            "payload_full": {"ok": True, "result_out_content": "report line one\nreport line two\n"},
            "payload_toc": {"ok": True, "result_out_toc": [{"title": "Results", "line": 1}]},
            "snapshot": "snapshot", "snapshot_hashes": replay.snapshot_hashes(self.snapshot),
            "report": {"path": "result.out", "sha256": replay.snapshot_hashes(self.snapshot)["result.out"]},
            "max_steps": 24, "execute_step": 22,
            "remaining_steps": 2,
        }
        self.save_checkpoint()

    def save_checkpoint(self):
        self.checkpoint_path.write_text(json.dumps(self.checkpoint))

    def run_branch(self, fake, *, condition="full", budget=None):
        return replay.run_branch(
            self.checkpoint_path, self.root / condition, condition, self.config,
            budget if budget is not None else replay.Budget(10), completion=fake,
        )

    def test_paired_messages_share_history_and_only_pending_payload_differs(self):
        before = copy.deepcopy(self.checkpoint)
        full = replay.build_messages(self.checkpoint, "full")
        toc = replay.build_messages(self.checkpoint, "toc")
        self.assertEqual(full[:-1], toc[:-1])
        self.assertEqual("system", full[0]["role"])
        self.assertEqual(self.checkpoint["messages"], full[1:-1])
        self.assertEqual("execute-1", full[-1]["tool_call_id"])
        self.assertEqual("execute-1", toc[-1]["tool_call_id"])
        self.assertEqual(self.checkpoint["payload_full"], json.loads(full[-1]["content"]))
        self.assertEqual(self.checkpoint["payload_toc"], json.loads(toc[-1]["content"]))
        self.assertNotIn("DO_NOT_SEND_SCORING_METADATA", json.dumps(full))
        full[1]["content"] = "mutated by fake model client"
        self.assertEqual(before, self.checkpoint)
        self.assertEqual("What is the pH?", toc[1]["content"])

    def test_both_arms_receive_same_read_tools_files_and_remaining_turns(self):
        captured = {}
        for arm in ("full", "toc"):
            captured[arm] = []

            def fake(**kwargs):
                captured[arm].append(copy.deepcopy(kwargs))
                if len(captured[arm]) == 1:
                    return tool_response("read_file", {"path": "result.out", "start_line": "2", "end_line": "2"})
                return tool_response()

            result = self.run_branch(fake, condition=arm)
            self.assertEqual("submitted", result["status"])
            self.assertEqual(2, result["calls"])
            self.assertTrue(result["is_correct"])
        self.assertEqual(captured["full"][0]["messages"][:-1], captured["toc"][0]["messages"][:-1])
        self.assertEqual(captured["full"][0]["tools"], captured["toc"][0]["tools"])
        self.assertEqual(captured["full"][1]["messages"][-1], captured["toc"][1]["messages"][-1])
        self.assertEqual(self.original, self.snapshot_bytes())

    def test_remaining_one_turn_stops_after_one_api_call(self):
        self.checkpoint["remaining_steps"] = 1
        self.checkpoint["execute_step"] = 23
        self.save_checkpoint()
        calls = []

        def fake(**kwargs):
            calls.append(kwargs)
            return tool_response("read_file", {"path": "result.out"})

        result = self.run_branch(fake)
        self.assertEqual(1, len(calls))
        self.assertEqual(1, result["calls"])
        self.assertEqual("max_steps", result["status"])
        self.assertIsNone(result["prediction"])

    def test_done_resume_never_calls_api_again(self):
        first = self.run_branch(lambda **kwargs: tool_response())

        def forbidden(**kwargs):
            self.fail("A completed branch must not bill another API request")

        second = self.run_branch(forbidden)
        self.assertEqual(first, second)

    def test_request_pending_resume_blocks_a_possibly_paid_duplicate(self):
        calls = []

        def interrupted(**kwargs):
            calls.append(kwargs)
            raise TimeoutError("response lost after request was sent")

        with self.assertRaises(TimeoutError):
            self.run_branch(interrupted)
        self.assertEqual("request_pending", json.loads((self.root / "full/state.json").read_text())["phase"])
        with self.assertRaisesRegex(RuntimeError, "paying twice"):
            self.run_branch(interrupted)
        self.assertEqual(1, len(calls))

    def test_persisted_response_finishes_without_calling_or_charging_twice(self):
        original_write = replay.atomic_json

        def interrupted_write(path, value):
            original_write(path, value)
            if path.name == "state.json" and value.get("phase") == "response_ready":
                raise InterruptedError("stopped after durable API response")

        budget = replay.Budget(10)
        with patch.object(replay, "atomic_json", side_effect=interrupted_write):
            with self.assertRaises(InterruptedError):
                self.run_branch(lambda **kwargs: tool_response(), budget=budget)
        charged = budget.spent_usd

        def forbidden(**kwargs):
            self.fail("Persisted response must be processed without another API call")

        result = self.run_branch(forbidden, budget=budget)
        self.assertEqual("submitted", result["status"])
        self.assertEqual(1, result["calls"])
        self.assertEqual(charged, budget.spent_usd)

    def test_budget_pause_resumes_from_saved_history_after_limit_increase(self):
        budget = replay.Budget(0.00001)
        calls = []

        def fake(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            if len(calls) == 1:
                return tool_response("read_file", {"path": "result.out"})
            return tool_response()

        paused = self.run_branch(fake, budget=budget)
        self.assertEqual("budget_paused", paused["status"])
        self.assertEqual(1, len(calls))
        state = json.loads((self.root / "full/state.json").read_text())
        self.assertEqual("ready", state["phase"])
        resumed_budget = replay.Budget(10, spent_usd=budget.spent_usd)
        result = self.run_branch(fake, budget=resumed_budget)
        self.assertEqual("submitted", result["status"])
        self.assertEqual(2, len(calls))
        self.assertEqual(2, result["calls"])
        self.assertIn("report line two", calls[1]["messages"][-1]["content"])
        self.assertAlmostEqual(result["estimated_cost_usd"], resumed_budget.spent_usd)

    def test_changed_frozen_files_are_rejected_before_api_call(self):
        (self.snapshot / "result.out").write_text("different computation")

        def forbidden(**kwargs):
            self.fail("Changed snapshots must fail before an API request")

        with self.assertRaisesRegex(ValueError, "hashes"):
            self.run_branch(forbidden)

    def test_malformed_checkpoint_contracts_fail_before_api_call(self):
        original = copy.deepcopy(self.checkpoint)
        broken = {}
        cp = copy.deepcopy(original)
        cp["schema_version"] = 999
        broken["unknown schema"] = cp
        cp = copy.deepcopy(original)
        cp["snapshot_hashes"] = {}
        broken["empty snapshot"] = cp
        cp = copy.deepcopy(original)
        cp["remaining_steps"] = 99
        broken["inflated remaining turns"] = cp
        cp = copy.deepcopy(original)
        cp["remaining_steps"] = True
        broken["boolean remaining turns"] = cp
        cp = copy.deepcopy(original)
        cp["report"]["sha256"] = "0" * 64
        broken["wrong report hash"] = cp
        cp = copy.deepcopy(original)
        cp["payload_toc"]["result_out_content"] = "unintended inline report"
        broken["inline evidence leaked into toc"] = cp
        cp = copy.deepcopy(original)
        cp["payload_toc"]["ok"] = False
        broken["different shared metadata"] = cp
        cp = copy.deepcopy(original)
        cp["execute_tool_call_id"] = "not-the-pending-id"
        broken["mismatched pending execution"] = cp
        cp = copy.deepcopy(original)
        cp["messages"].append({"role": "assistant", "content": "missing tool response"})
        broken["unanswered historical tool call"] = cp
        cp = copy.deepcopy(original)
        cp["messages"].insert(1, {"role": "tool", "tool_call_id": "orphan", "content": "{}"})
        broken["unmatched historical tool response"] = cp

        def forbidden(**kwargs):
            self.fail("Invalid checkpoints must fail before any API request")

        for name, cp in broken.items():
            with self.subTest(case=name):
                self.checkpoint = cp
                self.save_checkpoint()
                with self.assertRaises(ValueError):
                    self.run_branch(forbidden)
        self.checkpoint = original
        self.save_checkpoint()

    def test_exhausted_budget_never_starts_an_api_call(self):
        def forbidden(**kwargs):
            self.fail("An exhausted budget cannot start a paid request")

        result = self.run_branch(forbidden, budget=replay.Budget(1, spent_usd=1))
        self.assertEqual("budget_paused", result["status"])
        state = json.loads((self.root / "full/state.json").read_text())
        self.assertEqual(0, state["calls"])
        self.assertEqual("ready", state["phase"])

    def test_main_extends_one_repeat_to_three_without_rebilling_existing_pair(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1, "status": "complete",
            "eligible": [{"item_id": 2, "checkpoint": self.checkpoint_path.name}],
        }))
        output = self.root / "experiment"
        calls = []
        fake_litellm = types.ModuleType("litellm")

        def fake_completion(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            return tool_response()

        fake_litellm.completion = fake_completion
        base_argv = [
            "run_frozen_replay.py", "--manifest", str(manifest),
            "--output", str(output), "--workers", "1", "--run",
        ]

        def launch(*extra):
            # The fake marker only satisfies the CLI presence check. No real
            # credential or installed API client is read or used by this test.
            with patch.dict(sys.modules, {"litellm": fake_litellm}), \
                    patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-placeholder"}, clear=True), \
                    patch.object(sys, "argv", base_argv + list(extra)), \
                    redirect_stdout(io.StringIO()):
                replay.main()

        launch()  # Default must be one run of each condition.
        self.assertEqual(2, len(calls))
        first_states = {
            path: path.read_bytes()
            for path in (output / "runs" / "repeat_01").rglob("state.json")
        }
        self.assertEqual(2, len(first_states))
        self.assertEqual(1, json.loads((output / "experiment.json").read_text())["repeats"])

        launch("--repeats", "3")
        self.assertEqual(6, len(calls), "Only four additional branches may call the API")
        self.assertEqual(first_states, {path: path.read_bytes() for path in first_states})
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(6, summary["completed_branches"])
        self.assertEqual(3, summary["completed_pairs"])
        self.assertAlmostEqual(6 * (100 * 1.75 + 10 * 14) / 1_000_000,
                               summary["estimated_cost_usd"])
        self.assertEqual(3, json.loads((output / "experiment.json").read_text())["repeats"])

        for extra in (("--repeats", "2"),
                      ("--repeats", "3", "--max-completion-tokens", "1024")):
            with self.subTest(arguments=extra):
                with self.assertRaisesRegex(ValueError, "settings changed|repeat count decreased"):
                    launch(*extra)
                self.assertEqual(6, len(calls))
        self.assertEqual(3, json.loads((output / "experiment.json").read_text())["repeats"])


class BudgetTests(unittest.TestCase):
    def test_stop_prevents_new_requests_even_below_cost_limit(self):
        budget = replay.Budget(100)
        self.assertTrue(budget.available())
        budget.stop()
        self.assertFalse(budget.available())

    def test_concurrent_usage_accumulates_without_lost_updates(self):
        budget = replay.Budget(100)
        usage = {"prompt_tokens": 1000, "completion_tokens": 100}
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: budget.add(usage), range(1000)))
        self.assertAlmostEqual(3.15, budget.spent_usd, places=9)


if __name__ == "__main__":
    unittest.main()
