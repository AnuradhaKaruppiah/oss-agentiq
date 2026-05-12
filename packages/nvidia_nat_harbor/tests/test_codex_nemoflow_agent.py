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
"""Unit tests for the experimental Codex NeMo-Flow Harbor wrapper."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from harbor.models.agent.context import AgentContext

from nat_harbor.agents.installed.codex_nemoflow import CodexNeMoFlow


def _fake_nemo_flow_bin(tmp_path: Path) -> Path:
    binary = tmp_path / "nemo-flow"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _make_agent(tmp_path: Path, **kwargs) -> CodexNeMoFlow:
    logs_dir = tmp_path / "agent"
    logs_dir.mkdir(parents=True)
    return CodexNeMoFlow(
        logs_dir=logs_dir,
        model_name="nvidia/opus-frontier",
        nemo_flow_bin=str(_fake_nemo_flow_bin(tmp_path)),
        **kwargs,
    )


def test_build_provider_env_maps_nvidia_to_openai_compatible_codex_auth(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        extra_env={
            "NVIDIA_API_KEY": "nvidia-key",
            "NVIDIA_BASE_URL": "https://nvidia.example/v1",
        },
    )

    env, base_url = agent._provider_env_and_gateway_base_url()

    assert base_url == "https://nvidia.example/v1"
    assert env["NVIDIA_API_KEY"] == "nvidia-key"
    assert env["NVIDIA_BASE_URL"] == "https://nvidia.example/v1"
    assert env["OPENAI_API_KEY"] == "nvidia-key"


def test_codex_model_maps_nvidia_opus_alias(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)

    assert agent._codex_model() == "aws/anthropic/claude-opus-4-5"


def test_codex_model_kwarg_overrides_provider_alias(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, codex_model="custom/model")

    assert agent._codex_model() == "custom/model"


def test_populate_context_records_and_canonicalizes_gateway_atif(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    atif_dir = agent.logs_dir / "nemo-flow-atif"
    atif_dir.mkdir(parents=True)
    older = atif_dir / "older.atif.json"
    newer = atif_dir / "newer.atif.json"
    older.write_text('{"schema_version":"ATIF-v1.7","steps":[]}\n', encoding="utf-8")
    newer.write_text('{"schema_version":"ATIF-v1.7","steps":[{"step_id":1}]}\n', encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    context = AgentContext()

    agent.populate_context_post_run(context)

    canonical = atif_dir / "trajectory.json"
    assert canonical.read_text(encoding="utf-8") == newer.read_text(encoding="utf-8")
    assert context.metadata is not None
    assert context.metadata["nemo_flow_gateway_atif_paths"] == sorted([str(older), str(newer)])
    assert context.metadata["nemo_flow_gateway_canonical_atif_path"] == str(canonical)
    assert context.metadata["nemo_flow_gateway_canonical_atif_exists"] is True


def test_populate_context_can_fail_when_gateway_atif_missing(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, fail_missing_nemoflow_atif=True)

    with pytest.raises(FileNotFoundError, match="NeMo-Flow Codex ATIF"):
        agent.populate_context_post_run(AgentContext())


def test_missing_gateway_binary_fails_early(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NeMo-Flow gateway binary"):
        CodexNeMoFlow(
            logs_dir=tmp_path / "agent",
            model_name="nvidia/opus-frontier",
            nemo_flow_bin=str(tmp_path / "missing-nemo-flow"),
        )
