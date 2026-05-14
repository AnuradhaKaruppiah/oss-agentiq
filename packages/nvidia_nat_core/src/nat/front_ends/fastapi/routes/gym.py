# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym response route for NAT workflows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from nat.data_models.api_server import ResponseATIFStep
from nat.data_models.api_server import ResponseATIFTrajectory
from nat.data_models.api_server import ResponseObservabilityTrace
from nat.data_models.api_server import ResponsePayloadOutput
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_atif
from nat.runtime.session import SessionManager

from .common_utils import RESPONSE_500


class GymRunRequest(BaseModel):
    """Structurally compatible subset of a NeMo Gym rollout row."""

    model_config = ConfigDict(extra="allow")

    responses_create_params: dict[str, Any]


class GymRunResponse(BaseModel):
    """Structurally compatible subset of NeMo Gym BaseVerifyResponse."""

    model_config = ConfigDict(extra="allow")

    responses_create_params: dict[str, Any]
    response: dict[str, Any]
    reward: float
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    evaluator_details: dict[str, Any] = Field(default_factory=dict)
    nat_metadata: dict[str, Any] = Field(default_factory=dict)


class GymAggregateMetricsRequest(BaseModel):
    """Structurally compatible subset of NeMo Gym AggregateMetricsRequest."""

    verify_responses: list[dict[str, Any]]


class GymAggregateMetricsResponse(BaseModel):
    """Structurally compatible subset of NeMo Gym AggregateMetrics."""

    group_level_metrics: list[dict[str, Any]] = Field(default_factory=list)
    agent_metrics: dict[str, Any] = Field(default_factory=dict)
    key_metrics: dict[str, Any] = Field(default_factory=dict)


def _dump_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    return value


def _content_to_text(content: Any) -> str:
    content = _dump_jsonable(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            item = _dump_jsonable(item)
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(str(text) if text is not None else json.dumps(item, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        return str(text) if text is not None else json.dumps(content, sort_keys=True)
    return str(content)


def instruction_from_responses_create_params(params: dict[str, Any]) -> str:
    """Extract the user-facing NAT workflow input from Gym Responses params."""

    input_value = params.get("input")
    if isinstance(input_value, str):
        return input_value

    if isinstance(input_value, list):
        for message in reversed(input_value):
            message = _dump_jsonable(message)
            if isinstance(message, dict) and message.get("role") == "user":
                text = _content_to_text(message.get("content"))
                if text:
                    return text

        rendered_messages: list[str] = []
        for message in input_value:
            message = _dump_jsonable(message)
            if isinstance(message, dict):
                role = message.get("role", "message")
                text = _content_to_text(message.get("content"))
                if text:
                    rendered_messages.append(f"{role}: {text}")
            else:
                rendered_messages.append(str(message))
        return "\n".join(rendered_messages).strip()

    return _content_to_text(input_value)


def _payload_to_text(payload: Any) -> str:
    payload = _dump_jsonable(payload)
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("output", "value", "text", "content"):
            value = payload.get(key)
            if value is not None:
                return _content_to_text(value)
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                return _content_to_text(message.get("content"))
        return json.dumps(payload, sort_keys=True)
    return str(payload)


def _expected_output_obj(row: dict[str, Any]) -> Any:
    for key in ("expected_output_obj", "expected_output", "answer", "ground_truth", "rubric"):
        if key in row:
            return row[key]
    return None


def _item_id(row: dict[str, Any]) -> str:
    for key in ("item_id", "source_id", "id", "_ng_task_index"):
        if key in row:
            return str(row[key])
    return f"item-{uuid4().hex[:8]}"


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _evaluator_name(row: dict[str, Any]) -> str | None:
    metadata = _row_metadata(row)
    value = row.get("evaluator_name") or metadata.get("evaluator_name")
    return str(value) if value else None


def _default_artifact_dir(row: dict[str, Any]) -> Path:
    metadata = _row_metadata(row)
    configured = row.get("artifact_dir") or metadata.get("artifact_dir") or ".tmp/nat-gym/gym-run"
    return Path(str(configured)).expanduser() / f"run-{uuid4().hex[:12]}"


def _response_from_output_text(params: dict[str, Any], output_text: str) -> dict[str, Any]:
    return {
        "id": f"resp_{uuid4().hex}",
        "created_at": time.time(),
        "error": None,
        "incomplete_details": None,
        "instructions": params.get("instructions"),
        "metadata": params.get("metadata") or {},
        "model": params.get("model") or "nat-workflow",
        "object": "response",
        "output": [
            {
                "id": f"msg_{uuid4().hex}",
                "content": [
                    {
                        "annotations": [],
                        "text": output_text,
                        "type": "output_text",
                        "logprobs": None,
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": params.get("parallel_tool_calls", True),
        "temperature": params.get("temperature"),
        "tool_choice": params.get("tool_choice", "auto"),
        "tools": params.get("tools") or [],
        "top_p": params.get("top_p"),
        "background": params.get("background"),
        "max_output_tokens": params.get("max_output_tokens"),
        "max_tool_calls": params.get("max_tool_calls"),
        "previous_response_id": params.get("previous_response_id"),
        "prompt": params.get("prompt"),
        "reasoning": params.get("reasoning"),
        "service_tier": params.get("service_tier"),
        "status": "completed",
        "text": params.get("text"),
        "top_logprobs": params.get("top_logprobs"),
        "truncation": params.get("truncation"),
        "usage": None,
        "user": params.get("user"),
    }


async def _evaluate_row(
    *,
    worker: Any,
    row: dict[str, Any],
    instruction: str,
    output_text: str,
) -> tuple[float, dict[str, Any]]:
    evaluator_name = _evaluator_name(row)
    if evaluator_name is None:
        return float(row.get("reward", 0.0) or 0.0), {"mode": "not_configured"}

    if evaluator_name not in worker._evaluators:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluator '{evaluator_name}' not found. Available evaluators: {list(worker._evaluators.keys())}",
        )

    evaluator = worker._evaluators[evaluator_name]
    eval_item = EvalInputItem(
        id=_item_id(row),
        input_obj=instruction,
        expected_output_obj=_expected_output_obj(row),
        output_obj=output_text,
        full_dataset_entry=row,
    )
    eval_output = await evaluator.evaluate_fn(EvalInput(eval_input_items=[eval_item]))
    output_item = eval_output.eval_output_items[0] if eval_output.eval_output_items else None
    score = getattr(output_item, "score", None)
    if not isinstance(score, int | float):
        score = getattr(eval_output, "average_score", None)
    reward = float(score) if isinstance(score, int | float) else 0.0
    details = _dump_jsonable(output_item) if output_item is not None else {}
    return reward, {"mode": "evaluator", "evaluator_name": evaluator_name, "result": details}


def _write_artifacts(
    *,
    row: dict[str, Any],
    output_text: str,
    atif_trajectory: dict[str, Any],
    evaluator_details: dict[str, Any],
) -> dict[str, str]:
    artifact_dir = _default_artifact_dir(row)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    output_path = artifact_dir / "output.txt"
    trajectory_path = artifact_dir / "trajectory.json"
    evaluator_path = artifact_dir / "evaluator_details.json"

    output_path.write_text(output_text, encoding="utf-8")
    trajectory_path.write_text(json.dumps(atif_trajectory, indent=2, default=str), encoding="utf-8")
    evaluator_path.write_text(json.dumps(evaluator_details, indent=2, default=str), encoding="utf-8")

    return {
        "output_text": str(output_path),
        "trajectory": str(trajectory_path),
        "evaluator_details": str(evaluator_path),
    }


def _aggregate_metrics(verify_responses: list[dict[str, Any]]) -> GymAggregateMetricsResponse:
    rewards = [float(row["reward"]) for row in verify_responses if isinstance(row.get("reward"), int | float)]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    grouped: dict[str, list[float]] = {}
    for row in verify_responses:
        reward = row.get("reward")
        if not isinstance(reward, int | float):
            continue
        task_id = row.get("_ng_task_index", row.get("task_index", row.get("id", "unknown")))
        grouped.setdefault(str(task_id), []).append(float(reward))

    group_level_metrics = [
        {
            "_ng_task_index": task_id,
            "num_rollouts": len(task_rewards),
            "mean_reward": sum(task_rewards) / len(task_rewards),
        }
        for task_id, task_rewards in sorted(grouped.items())
    ]
    agent_metrics = {
        "num_rollouts": len(verify_responses),
        "num_scored_rollouts": len(rewards),
        "mean_reward": mean_reward,
    }
    return GymAggregateMetricsResponse(
        group_level_metrics=group_level_metrics,
        agent_metrics=agent_metrics,
        key_metrics={
            "mean_reward": mean_reward,
        },
    )


async def add_gym_routes(worker: Any, app: FastAPI, session_manager: SessionManager):
    """Register NAT-to-NeMo-Gym bridge routes."""

    endpoint = worker.front_end_config.gym
    if not endpoint.path:
        return

    async def run_gym_row(body: GymRunRequest, http_request: Request) -> GymRunResponse:
        row = body.model_dump(mode="json")
        params = row["responses_create_params"]
        instruction = instruction_from_responses_create_params(params)

        atif_steps: list[dict[str, Any]] = []
        atif_summary: dict[str, Any] = {}
        observability_trace_id: str | None = None
        final_payload: Any = None

        async with session_manager.session(http_connection=http_request) as session:
            response_type = (
                session.workflow.streaming_output_schema
                if session.workflow.has_streaming_output
                else session.workflow.single_output_schema
            )
            async for item in generate_streaming_response_atif(
                instruction,
                session=session,
                streaming=True,
                result_type=response_type,
                output_type=response_type,
            ):
                if isinstance(item, ResponseATIFStep):
                    atif_steps.append(item.model_dump(mode="json", exclude_none=True))
                elif isinstance(item, ResponseATIFTrajectory):
                    atif_summary = item.model_dump(mode="json", exclude_none=True)
                elif isinstance(item, ResponseObservabilityTrace):
                    observability_trace_id = item.observability_trace_id
                elif isinstance(item, ResponsePayloadOutput):
                    final_payload = item.payload

        output_text = _payload_to_text(final_payload)
        response = _response_from_output_text(params, output_text)

        atif_trajectory = {
            "schema_version": atif_summary.get("schema_version", "ATIF-v1.7"),
            "session_id": atif_summary.get("session_id", f"session-{uuid4().hex}"),
            "agent": atif_summary.get("agent", {"name": "nat-agent"}),
            "steps": atif_steps,
        }
        if atif_summary.get("final_metrics"):
            atif_trajectory["final_metrics"] = atif_summary["final_metrics"]

        reward, evaluator_details = await _evaluate_row(
            worker=worker,
            row=row,
            instruction=instruction,
            output_text=output_text,
        )
        artifact_refs = _write_artifacts(
            row=row,
            output_text=output_text,
            atif_trajectory=atif_trajectory,
            evaluator_details=evaluator_details,
        )

        return GymRunResponse.model_validate(
            row
            | {
                "response": response,
                "reward": reward,
                "artifact_refs": artifact_refs,
                "evaluator_details": evaluator_details,
                "nat_metadata": {
                    "observability_trace_id": observability_trace_id,
                },
            }
        )

    def _add_run_route(path: str) -> None:
        app.add_api_route(
            path=path,
            endpoint=run_gym_row,
            methods=[endpoint.method],
            response_model=GymRunResponse,
            description=endpoint.description,
            responses={500: RESPONSE_500},
        )

    _add_run_route(endpoint.path)
    if endpoint.legacy_path and endpoint.legacy_path != endpoint.path:
        _add_run_route(endpoint.legacy_path)

    async def aggregate_metrics(body: GymAggregateMetricsRequest) -> GymAggregateMetricsResponse:
        return _aggregate_metrics(body.verify_responses)

    app.add_api_route(
        path="/aggregate_metrics",
        endpoint=aggregate_metrics,
        methods=["POST"],
        response_model=GymAggregateMetricsResponse,
        description="Aggregate NeMo Gym rollout metrics for NAT-backed Gym rows.",
        responses={500: RESPONSE_500},
    )
