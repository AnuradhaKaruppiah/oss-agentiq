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
"""Tests for NAT config/plugin runtime profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nat_harbor.capability_profiles import AgentHarnessRuntimeConfig
from nat_harbor.capability_profiles import deep_merge_dicts
from nat_harbor.capability_profiles import load_agent_harness_config


def _write_config(path: Path) -> Path:
    path.write_text(
        """
functions:
  current_timezone:
    _type: current_timezone

function_groups:
  mcp_calculator:
    _type: mcp_client
    server:
      transport: streamable-http
      url: http://host.docker.internal:9901/mcp
    include:
      - calculator__add
      - calculator__multiply
      - calculator__compare

agent_harnesses:
  opencode_base:
    _type: opencode
    tool_names: []

  opencode_with_calculator_mcp:
    _type: opencode
    tool_names:
      - mcp_calculator
    skills_dir: /workspace/skills/debugging
    opencode_config:
      experimental:
        continue_loop_on_deny: true
    env:
      NAT_PROFILE_ID: opencode_with_calculator_mcp

eval:
  general:
    max_concurrency: 1
    output:
      dir: .tmp/nat/harbor/calculator-mcp-smoke
      write_atif_workflow_output: true
    dataset:
      _type: harbor
      path: packages/nvidia_nat_harbor/examples/tasks/calculator-mcp-smoke
    agent_harnesses:
      - name: baseline
        agent_harness: opencode_base
      - name: with_calculator_mcp
        agent_harness: opencode_with_calculator_mcp
  evaluators:
    task_verifier:
      _type: harbor_task_verifier
      affects_reward: true
""",
        encoding="utf-8",
    )
    return path


def test_load_nat_style_config_and_apply_agent_harness(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yml")

    applied = load_agent_harness_config(config_path).apply(
        agent_harness="opencode_with_calculator_mcp",
        source_path=config_path,
    )

    assert applied.name == "opencode_with_calculator_mcp"
    assert applied.tool_names == ("mcp_calculator",)
    assert applied.mcp_function_groups == ("mcp_calculator",)
    assert applied.mcp_tool_filters == {
        "mcp_calculator": {
            "include": [
                "calculator__add",
                "calculator__multiply",
                "calculator__compare",
            ]
        }
    }
    assert [server.name for server in applied.mcp_servers] == ["mcp_calculator"]
    assert applied.mcp_servers[0].transport == "streamable-http"
    assert applied.mcp_servers[0].url == "http://host.docker.internal:9901/mcp"
    assert applied.skills_dir == "/workspace/skills/debugging"
    assert applied.env == {"NAT_PROFILE_ID": "opencode_with_calculator_mcp"}
    assert applied.opencode_config["experimental"]["continue_loop_on_deny"] is True
    assert applied.metadata()["source_path"] == str(config_path)
    assert applied.metadata()["mcp_tool_filters"] == applied.mcp_tool_filters


def test_eval_general_agent_harnesses_define_profile_variants(tmp_path: Path) -> None:
    config = load_agent_harness_config(_write_config(tmp_path / "config.yml"))

    runs = config.eval_agent_harness_runs()

    assert [(run.name, run.agent_harness) for run in runs] == [
        ("baseline", "opencode_base"),
        ("with_calculator_mcp", "opencode_with_calculator_mcp"),
    ]
    assert config.eval is not None
    assert config.eval.general is not None
    assert config.eval.general.dataset is not None
    assert config.eval.general.dataset.dataset_type == "harbor"
    assert config.eval.general.dataset.path == "packages/nvidia_nat_harbor/examples/tasks/calculator-mcp-smoke"
    assert config.eval.evaluators["task_verifier"]["affects_reward"] is True


def test_load_config_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({
            "function_groups": {
                "mcp_calculator": {
                    "_type": "mcp_client",
                    "server": {
                        "transport": "streamable-http",
                        "url": "http://host.docker.internal:9901/mcp",
                    },
                }
            },
            "agent_harnesses": {
                "opencode_with_calculator_mcp": {
                    "_type": "opencode",
                    "tool_names": ["mcp_calculator"],
                }
            },
        }),
        encoding="utf-8",
    )

    applied = load_agent_harness_config(config_path).apply(agent_harness="opencode_with_calculator_mcp")

    assert [server.name for server in applied.mcp_servers] == ["mcp_calculator"]


def test_single_agent_harness_can_be_inferred(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
function_groups:
  mcp_calculator:
    _type: mcp_client
    server:
      transport: streamable-http
      url: http://host.docker.internal:9901/mcp
agent_harnesses:
  opencode_with_calculator_mcp:
    _type: opencode
    tool_names: [mcp_calculator]
""",
        encoding="utf-8",
    )

    applied = load_agent_harness_config(config_path).apply()

    assert applied.name == "opencode_with_calculator_mcp"


def test_local_nat_functions_are_rejected_for_opencode_poc() -> None:
    config = AgentHarnessRuntimeConfig.model_validate({
        "functions": {
            "current_timezone": {
                "_type": "current_timezone"
            }
        },
        "agent_harnesses": {
            "opencode_with_function": {
                "_type": "opencode",
                "tool_names": ["current_timezone"],
            }
        },
    })

    with pytest.raises(ValueError, match="supports MCP function_groups only"):
        config.apply(agent_harness="opencode_with_function")


def test_deep_merge_dicts_does_not_mutate_inputs() -> None:
    base = {"provider": {"nvidia": {"models": {"a": {}}}}}
    override = {"provider": {"nvidia": {"options": {"baseURL": "http://example"}}}}

    merged = deep_merge_dicts(base, override)

    assert merged == {
        "provider": {
            "nvidia": {
                "models": {
                    "a": {}
                },
                "options": {
                    "baseURL": "http://example"
                },
            }
        }
    }
    assert base == {"provider": {"nvidia": {"models": {"a": {}}}}}
