# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.routes.gym import _aggregate_metrics
from nat.front_ends.fastapi.routes.gym import _payload_to_text
from nat.front_ends.fastapi.routes.gym import _response_from_output_text
from nat.front_ends.fastapi.routes.gym import _write_artifacts
from nat.front_ends.fastapi.routes.gym import instruction_from_responses_create_params


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
