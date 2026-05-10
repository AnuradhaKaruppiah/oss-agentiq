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
"""Tests for AgentEvals ATIF trajectory judge smoke utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nat_harbor.smoke import judge_atif_trajectories
from nat_harbor.smoke.judge_atif_trajectories import CANDIDATE_HIGHER
from nat_harbor.smoke.judge_atif_trajectories import ERROR
from nat_harbor.smoke.judge_atif_trajectories import SCORE_SAME
from nat_harbor.smoke.judge_atif_trajectories import AgentevalsJudgeLoader
from nat_harbor.smoke.judge_atif_trajectories import JudgeConfig
from nat_harbor.smoke.judge_atif_trajectories import judge_trial
from nat_harbor.smoke.judge_atif_trajectories import main


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _trajectory(*, final_message: str) -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-1",
        "agent": {
            "name": "opencode",
            "version": "test",
        },
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": "Fix the failing test.",
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": final_message,
            },
        ],
    }


def _write_trial(tmp_path: Path, *, native_message: str, candidate_message: str) -> Path:
    trial_dir = tmp_path / "django__django-13741__abc123"
    _write_json(trial_dir / "agent" / "trajectory.json", _trajectory(final_message=native_message))
    _write_json(
        trial_dir / "agent" / "nemo-flow-atof-atif" / "trajectory.json",
        _trajectory(final_message=candidate_message),
    )
    _write_json(trial_dir / "result.json", {"verifier_result": {"rewards": {"reward": 1.0}}})
    return trial_dir


def _messages(trajectory: dict) -> list[dict]:
    return [{"role": step["source"], "content": step["message"]} for step in trajectory["steps"]]


def _fake_judge(*, outputs):
    final_message = outputs[-1]["content"]
    return {
        "key": "trajectory_accuracy",
        "score": "better" in final_message,
        "comment": final_message,
    }


def test_judge_trial_scores_native_and_candidate_independently(tmp_path: Path) -> None:
    trial_dir = _write_trial(tmp_path, native_message="bad answer", candidate_message="better answer")

    row = judge_trial(
        trial_dir,
        native_rel=Path("agent/trajectory.json"),
        candidate_rel=Path("agent/nemo-flow-atof-atif/trajectory.json"),
        judge_config=JudgeConfig(model="openai:test", continuous=False, score_threshold=0.05),
        atif_to_openai_messages=_messages,
        judge=_fake_judge,
    )

    assert row.task_id == "django__django-13741"
    assert row.swebench_reward == pytest.approx(1.0)
    assert row.native_score is False
    assert row.candidate_score is True
    assert row.score_delta == pytest.approx(1.0)
    assert row.score_category == CANDIDATE_HIGHER
    assert row.native_comment == "bad answer"
    assert row.candidate_comment == "better answer"


def test_judge_trial_reports_score_same_within_threshold(tmp_path: Path) -> None:
    trial_dir = _write_trial(tmp_path, native_message="better answer", candidate_message="better answer")

    row = judge_trial(
        trial_dir,
        native_rel=Path("agent/trajectory.json"),
        candidate_rel=Path("agent/nemo-flow-atof-atif/trajectory.json"),
        judge_config=JudgeConfig(model="openai:test", continuous=False, score_threshold=0.05),
        atif_to_openai_messages=_messages,
        judge=_fake_judge,
    )

    assert row.score_delta == pytest.approx(0.0)
    assert row.score_category == SCORE_SAME


def test_judge_trial_keeps_batch_resilient_on_conversion_error(tmp_path: Path) -> None:
    trial_dir = _write_trial(tmp_path, native_message="native", candidate_message="candidate")

    def _failing_converter(_trajectory):
        raise ValueError("bad ATIF")

    row = judge_trial(
        trial_dir,
        native_rel=Path("agent/trajectory.json"),
        candidate_rel=Path("agent/nemo-flow-atof-atif/trajectory.json"),
        judge_config=JudgeConfig(model="openai:test", continuous=False, score_threshold=0.05),
        atif_to_openai_messages=_failing_converter,
        judge=_fake_judge,
    )

    assert row.score_category == ERROR
    assert row.native_error == "ValueError: bad ATIF"
    assert row.candidate_error == "ValueError: bad ATIF"


def test_main_writes_csv_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_trial(tmp_path, native_message="bad answer", candidate_message="better answer")
    output_dir = tmp_path / "reports"

    def _fake_loader(_agentevals_python_path: Path | None) -> AgentevalsJudgeLoader:
        def _factory(**_kwargs):
            return _fake_judge

        return AgentevalsJudgeLoader(
            atif_to_openai_messages=_messages,
            create_trajectory_llm_as_judge=_factory,
            trajectory_accuracy_prompt="prompt",
        )

    monkeypatch.setattr(judge_atif_trajectories, "_load_agentevals", _fake_loader)
    monkeypatch.setattr(
        "sys.argv",
        [
            "judge_atif_trajectories",
            "--job-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--model",
            "openai:test",
        ],
    )

    main()

    csv_text = (output_dir / "atif-trajectory-judge.csv").read_text(encoding="utf-8")
    markdown_text = (output_dir / "atif-trajectory-judge.md").read_text(encoding="utf-8")
    assert "django__django-13741" in csv_text
    assert "candidate_higher" in csv_text
    assert "- candidate_higher: 1" in markdown_text
