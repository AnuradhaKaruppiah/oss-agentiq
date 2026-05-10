# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for AgentEvals ATIF trajectory matching smoke utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nat_harbor.smoke import match_atif_trajectories
from nat_harbor.smoke.match_atif_trajectories import ERROR
from nat_harbor.smoke.match_atif_trajectories import MATCH
from nat_harbor.smoke.match_atif_trajectories import MISMATCH
from nat_harbor.smoke.match_atif_trajectories import AgentevalsLoader
from nat_harbor.smoke.match_atif_trajectories import MatchConfig
from nat_harbor.smoke.match_atif_trajectories import main
from nat_harbor.smoke.match_atif_trajectories import match_trial


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _trajectory(*, tools: list[str], final_message: str = "Done.") -> dict:
    steps = [
        {
            "step_id": 1,
            "source": "user",
            "message": "Fix the failing test.",
        }
    ]
    for step_id, tool in enumerate(tools, start=2):
        steps.append({
            "step_id": step_id,
            "source": "agent",
            "message": "(tool use)",
            "tool_calls": [{
                "tool_call_id": f"call-{step_id}",
                "function_name": tool,
                "arguments": {
                    "path": "file.py"
                },
            }],
        })
    steps.append({
        "step_id": len(steps) + 1,
        "source": "agent",
        "message": final_message,
    })
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-1",
        "agent": {
            "name": "opencode",
            "version": "test",
        },
        "steps": steps,
    }


def _write_trial(tmp_path: Path, *, native_tools: list[str], candidate_tools: list[str]) -> Path:
    trial_dir = tmp_path / "django__django-13741__abc123"
    _write_json(trial_dir / "agent" / "trajectory.json", _trajectory(tools=native_tools))
    _write_json(
        trial_dir / "agent" / "nemo-flow-atof-atif" / "trajectory.json",
        _trajectory(tools=candidate_tools),
    )
    _write_json(trial_dir / "result.json", {"verifier_result": {"rewards": {"reward": 1.0}}})
    return trial_dir


def _openai_tool_names(trajectory: dict) -> list[str]:
    names: list[str] = []
    for step in trajectory["steps"]:
        for tool_call in step.get("tool_calls", []):
            names.append(tool_call["function_name"])
    return names


def _fake_evaluator(*, outputs, reference_outputs):
    return {
        "key": "trajectory_strict_match",
        "score": outputs == reference_outputs,
        "comment": None,
        "metadata": None,
    }


def test_match_trial_reports_match_for_equivalent_trajectories(tmp_path: Path) -> None:
    trial_dir = _write_trial(tmp_path, native_tools=["read", "edit"], candidate_tools=["read", "edit"])

    row = match_trial(
        trial_dir,
        native_rel=Path("agent/trajectory.json"),
        candidate_rel=Path("agent/nemo-flow-atof-atif/trajectory.json"),
        match_config=MatchConfig(trajectory_match_mode="strict", tool_args_match_mode="exact"),
        atif_to_openai_messages=_openai_tool_names,
        evaluator=_fake_evaluator,
    )

    assert row.task_id == "django__django-13741"
    assert row.swebench_reward == pytest.approx(1.0)
    assert row.deterministic_comparison == "match (same)"
    assert row.evaluator_key == "trajectory_strict_match"
    assert row.evaluator_score is True
    assert row.match_category == MATCH
    assert row.error is None


def test_match_trial_reports_mismatch_for_different_trajectories(tmp_path: Path) -> None:
    trial_dir = _write_trial(tmp_path, native_tools=["read", "edit"], candidate_tools=["glob", "read", "edit"])

    row = match_trial(
        trial_dir,
        native_rel=Path("agent/trajectory.json"),
        candidate_rel=Path("agent/nemo-flow-atof-atif/trajectory.json"),
        match_config=MatchConfig(trajectory_match_mode="strict", tool_args_match_mode="exact"),
        atif_to_openai_messages=_openai_tool_names,
        evaluator=_fake_evaluator,
    )

    assert row.deterministic_comparison == "match (richer)"
    assert row.evaluator_score is False
    assert row.match_category == MISMATCH


def test_match_trial_keeps_batch_resilient_on_conversion_error(tmp_path: Path) -> None:
    trial_dir = _write_trial(tmp_path, native_tools=["read"], candidate_tools=["read"])

    def _failing_converter(_trajectory):
        raise ValueError("bad ATIF")

    row = match_trial(
        trial_dir,
        native_rel=Path("agent/trajectory.json"),
        candidate_rel=Path("agent/nemo-flow-atof-atif/trajectory.json"),
        match_config=MatchConfig(trajectory_match_mode="strict", tool_args_match_mode="exact"),
        atif_to_openai_messages=_failing_converter,
        evaluator=_fake_evaluator,
    )

    assert row.match_category == ERROR
    assert row.error == "ValueError: bad ATIF"


def test_main_writes_csv_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_trial(tmp_path, native_tools=["read"], candidate_tools=["read"])
    output_dir = tmp_path / "reports"

    def _fake_loader(_agentevals_python_path: Path | None) -> AgentevalsLoader:
        def _factory(**_kwargs):
            return _fake_evaluator

        return AgentevalsLoader(
            atif_to_openai_messages=_openai_tool_names,
            create_trajectory_match_evaluator=_factory,
        )

    monkeypatch.setattr(match_atif_trajectories, "_load_agentevals", _fake_loader)
    monkeypatch.setattr(
        "sys.argv",
        [
            "match_atif_trajectories",
            "--job-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    csv_text = (output_dir / "atif-trajectory-match.csv").read_text(encoding="utf-8")
    markdown_text = (output_dir / "atif-trajectory-match.md").read_text(encoding="utf-8")
    assert "django__django-13741" in csv_text
    assert "trajectory_strict_match" in csv_text
    assert "- match: 1" in markdown_text
