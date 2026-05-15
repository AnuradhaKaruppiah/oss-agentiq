# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from pydantic import Field

from nat.data_models.api_server import ResponseATIFTrajectory
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.routes.gym import _aggregate_metrics
from nat.front_ends.fastapi.routes.gym import _evaluate_row
from nat.front_ends.fastapi.routes.gym import _payload_to_text
from nat.front_ends.fastapi.routes.gym import _response_from_output_text
from nat.front_ends.fastapi.routes.gym import _write_artifacts
from nat.front_ends.fastapi.routes.gym import instruction_from_responses_create_params


@pytest.fixture(name="auto_set_env_vars", autouse=True)
def fixture_auto_set_env_vars():
    return


@pytest.fixture(autouse=True)
def patch_job_store_get_dask_client():
    yield


class _FakeEvalOutputItem(BaseModel):
    id: str
    score: float
    reasoning: dict = Field(default_factory=dict)
    error: str | None = None


def _fake_eval_output(score: float = 0.75):
    return SimpleNamespace(eval_output_items=[_FakeEvalOutputItem(id="task-1", score=score)], average_score=score)


def test_fastapi_config_exposes_gym_endpoint_by_default():
    config = FastApiFrontEndConfig()

    assert config.gym.path == "/v1/gym/run"
    assert config.gym.legacy_path == "/run"
    assert config.gym.method == "POST"


def test_instruction_from_responses_create_params_prefers_last_user_message():
    params = {
        "input": [
            {
                "role": "system",
                "content": "Be concise.",
            },
            {
                "role": "user",
                "content": "First question",
            },
            {
                "role": "assistant",
                "content": "Earlier answer",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Final question",
                    }
                ],
            },
        ]
    }

    assert instruction_from_responses_create_params(params) == "Final question"


def test_payload_to_text_handles_generate_response_dict():
    assert _payload_to_text({"output": "done", "value": "ignored"}) == "done"


def test_response_from_output_text_is_responses_api_shaped():
    response = _response_from_output_text(
        {
            "input": "hello",
            "model": "nat-test-model",
            "metadata": {
                "dataset": "smoke",
            },
        },
        "world",
    )

    assert response["object"] == "response"
    assert response["model"] == "nat-test-model"
    assert response["metadata"] == {"dataset": "smoke"}
    assert response["output"][0]["content"][0]["text"] == "world"


def test_response_atif_trajectory_allows_missing_session_id():
    summary = ResponseATIFTrajectory(schema_version="ATIF-v1.7", agent={"name": "nat-agent"})

    assert summary.session_id is None
    assert "session_id" not in summary.model_dump(exclude_none=True)


def test_write_artifacts_uses_row_artifact_dir(tmp_path):
    refs = _write_artifacts(
        row={
            "artifact_dir": str(tmp_path),
        },
        output_text="answer",
        atif_trajectory={
            "steps": [],
        },
        evaluator_details={
            "score": 1.0,
        },
    )

    assert refs["output_text"].startswith(str(tmp_path))
    assert refs["trajectory"].startswith(str(tmp_path))
    assert refs["evaluator_details"].startswith(str(tmp_path))
    assert json.loads(Path(refs["trajectory"]).read_text(encoding="utf-8")) == {"steps": []}


@pytest.mark.asyncio
async def test_evaluate_row_uses_legacy_evaluator():
    evaluator = SimpleNamespace(evaluate_fn=AsyncMock(return_value=_fake_eval_output(0.6)))
    worker = SimpleNamespace(_evaluators={"tuneable_eval": evaluator})

    reward, details = await _evaluate_row(
        worker=worker,
        row={
            "id": "task-1",
            "evaluator_name": "tuneable_eval",
            "expected_output_obj": "expected",
        },
        instruction="question",
        output_text="answer",
        atif_trajectory=None,
    )

    assert reward == 0.6
    assert details["mode"] == "evaluator"
    assert details["evaluator_name"] == "tuneable_eval"
    evaluator.evaluate_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluate_row_prefers_atif_evaluator_when_available():
    evaluator = SimpleNamespace(
        evaluate_fn=AsyncMock(return_value=_fake_eval_output(0.0)),
        evaluate_atif_fn=AsyncMock(return_value=_fake_eval_output(0.9)),
    )
    worker = SimpleNamespace(_evaluators={"trajectory_eval": evaluator})

    reward, details = await _evaluate_row(
        worker=worker,
        row={
            "id": "task-1",
            "evaluator_name": "trajectory_eval",
            "expected_output_obj": "expected",
        },
        instruction="question",
        output_text="answer",
        atif_trajectory={
            "schema_version": "ATIF-v1.7",
            "session_id": "session-1",
            "agent": {
                "name": "nat-agent",
                "version": "0.0.0",
            },
            "steps": [
                {
                    "step_id": 1,
                    "source": "user",
                    "message": "question",
                },
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "answer",
                },
            ],
        },
    )

    assert reward == 0.9
    assert details["mode"] == "atif_evaluator"
    assert details["evaluator_name"] == "trajectory_eval"
    evaluator.evaluate_atif_fn.assert_awaited_once()
    evaluator.evaluate_fn.assert_not_called()


def test_aggregate_metrics_returns_gym_compatible_shape():
    metrics = _aggregate_metrics(
        [
            {
                "_ng_task_index": 0,
                "reward": 1.0,
            },
            {
                "_ng_task_index": 0,
                "reward": 0.0,
            },
            {
                "_ng_task_index": 1,
                "reward": 1.0,
            },
        ]
    )

    assert metrics.agent_metrics["num_rollouts"] == 3
    assert metrics.agent_metrics["mean_reward"] == 2 / 3
    assert metrics.key_metrics["mean_reward"] == 2 / 3
    assert metrics.group_level_metrics[0]["mean_reward"] == 0.5
