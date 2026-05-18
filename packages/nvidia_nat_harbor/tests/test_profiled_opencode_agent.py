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


def _write_profile(path: Path) -> Path:
    path.write_text(
        """
name: profiled-tools
capabilities:
  - name: code-search
    kind: mcp_server
    mcp_server:
      name: code-search
      transport: stdio
      command: python
      args:
        - -m
        - code_search_mcp
  - name: debug-skills
    kind: skills_dir
    skills_dir: /workspace/skills/debugging
  - name: opencode-overlay
    kind: opencode_config
    opencode_config:
      experimental:
        continue_loop_on_deny: true
      provider:
        nvidia:
          options:
            baseURL: "{env:NVIDIA_BASE_URL}"
  - name: profile-env
    kind: env
    env:
      NAT_PROFILE_ID: profiled-tools
""",
        encoding="utf-8",
    )
    return path


def _make_agent(tmp_path: Path, **kwargs) -> ProfiledOpenCode:
    logs_dir = tmp_path / "agent"
    logs_dir.mkdir(parents=True)
    return ProfiledOpenCode(logs_dir=logs_dir, model_name="nvidia/opus-frontier", **kwargs)


def test_profiled_opencode_applies_profile_surfaces(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.yml")
    existing_mcp = MCPServerConfig(name="existing", transport="sse", url="http://existing.example/sse")

    agent = _make_agent(
        tmp_path,
        capability_profile=profile_path,
        mcp_servers=[existing_mcp],
        extra_env={"NAT_PROFILE_ID": "explicit-wins", "OTHER": "1"},
    )

    assert [server.name for server in agent.mcp_servers] == ["existing", "code-search"]
    assert agent.skills_dir == "/workspace/skills/debugging"
    assert agent._extra_env == {"NAT_PROFILE_ID": "explicit-wins", "OTHER": "1"}
    assert agent._opencode_config["experimental"]["continue_loop_on_deny"] is True
    assert agent._opencode_config["provider"]["nvidia"]["options"]["baseURL"] == "{env:NVIDIA_BASE_URL}"


def test_profiled_opencode_preserves_explicit_skills_and_config(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.yml")

    agent = _make_agent(
        tmp_path,
        capability_profile=profile_path,
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
    profile_path = _write_profile(tmp_path / "profile.yml")
    agent = _make_agent(tmp_path, capability_profile=profile_path)
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.metadata is not None
    profile_metadata = context.metadata["nat_capability_profile"]
    assert profile_metadata["name"] == "profiled-tools"
    assert profile_metadata["source_path"] == str(profile_path)
    assert [capability["name"] for capability in profile_metadata["enabled_capabilities"]] == [
        "code-search",
        "debug-skills",
        "opencode-overlay",
        "profile-env",
    ]


def test_profiled_opencode_uses_environment_profile_fallback(tmp_path: Path, monkeypatch) -> None:
    profile_path = _write_profile(tmp_path / "profile.yml")
    monkeypatch.setenv("NAT_HARBOR_CAPABILITY_PROFILE", str(profile_path))

    agent = _make_agent(tmp_path)

    assert [server.name for server in agent.mcp_servers] == ["code-search"]
