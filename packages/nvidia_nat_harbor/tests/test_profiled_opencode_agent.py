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
"""Tests for the profiled OpenCode Harbor wrapper."""

from __future__ import annotations

from pathlib import Path

from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from nat_harbor.agents.installed.profiled_opencode import ProfiledOpenCode


def _write_config(path: Path) -> Path:
    path.write_text(
        """
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
  opencode_with_calculator_mcp:
    _type: opencode
    tool_names:
      - mcp_calculator
    skills_dir: /workspace/skills/debugging
    opencode_config:
      experimental:
        continue_loop_on_deny: true
      provider:
        nvidia:
          options:
            baseURL: "{env:NVIDIA_BASE_URL}"
    env:
      NAT_PROFILE_ID: opencode_with_calculator_mcp
""",
        encoding="utf-8",
    )
    return path


def _make_agent(tmp_path: Path, **kwargs) -> ProfiledOpenCode:
    logs_dir = tmp_path / "agent"
    logs_dir.mkdir(parents=True)
    return ProfiledOpenCode(logs_dir=logs_dir, model_name="nvidia/explicit-model", **kwargs)


def test_profiled_opencode_applies_nat_config_harness_surfaces(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yml")
    existing_mcp = MCPServerConfig(name="existing", transport="sse", url="http://existing.example/sse")

    agent = _make_agent(
        tmp_path,
        config_file=config_path,
        agent_harness="opencode_with_calculator_mcp",
        mcp_servers=[existing_mcp],
        extra_env={"NAT_PROFILE_ID": "explicit-wins", "OTHER": "1"},
    )

    assert [server.name for server in agent.mcp_servers] == ["existing", "mcp_calculator"]
    assert agent.skills_dir == "/workspace/skills/debugging"
    assert agent.model_name == "nvidia/explicit-model"
    assert agent._extra_env == {"NAT_PROFILE_ID": "explicit-wins", "OTHER": "1"}
    assert agent._opencode_config["experimental"]["continue_loop_on_deny"] is True
    assert agent._opencode_config["provider"]["nvidia"]["options"]["baseURL"] == "{env:NVIDIA_BASE_URL}"


def test_profiled_opencode_keeps_model_as_run_level_argument(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yml")
    logs_dir = tmp_path / "agent"
    logs_dir.mkdir(parents=True)

    agent = ProfiledOpenCode(logs_dir=logs_dir, config_file=config_path, agent_harness="opencode_with_calculator_mcp")

    assert agent.model_name is None


def test_profiled_opencode_preserves_explicit_skills_and_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yml")

    agent = _make_agent(
        tmp_path,
        config_file=config_path,
        agent_harness="opencode_with_calculator_mcp",
        skills_dir="/task/skills",
        opencode_config={
            "experimental": {
                "continue_loop_on_deny": False
            }
        },
    )

    assert agent.skills_dir == "/task/skills"
    assert agent._opencode_config["experimental"]["continue_loop_on_deny"] is False


def test_profiled_opencode_records_profile_metadata(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yml")
    agent = _make_agent(tmp_path, config_file=config_path, agent_harness="opencode_with_calculator_mcp")
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.metadata is not None
    profile_metadata = context.metadata["nat_agent_harness_profile"]
    assert profile_metadata["agent_harness"] == "opencode_with_calculator_mcp"
    assert profile_metadata["source_path"] == str(config_path)
    assert profile_metadata["tool_names"] == ["mcp_calculator"]
    assert profile_metadata["mcp_function_groups"] == ["mcp_calculator"]
    assert profile_metadata["mcp_tool_filters"] == {
        "mcp_calculator": {
            "include": [
                "calculator__add",
                "calculator__multiply",
                "calculator__compare",
            ]
        }
    }


def test_profiled_opencode_uses_environment_profile_fallback(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path / "config.yml")
    monkeypatch.setenv("NAT_HARBOR_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("NAT_HARBOR_AGENT_HARNESS", "opencode_with_calculator_mcp")

    agent = _make_agent(tmp_path)

    assert [server.name for server in agent.mcp_servers] == ["mcp_calculator"]
