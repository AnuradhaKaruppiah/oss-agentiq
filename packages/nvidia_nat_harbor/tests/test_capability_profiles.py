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
"""Tests for NAT runtime capability profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nat_harbor.capability_profiles import CapabilityProfile
from nat_harbor.capability_profiles import deep_merge_dicts
from nat_harbor.capability_profiles import load_capability_profile


def _write_profile(path: Path) -> Path:
    path.write_text(
        """
schema_version: nat.capability_profile.v1
name: code-intelligence
description: Enable code search without changing the benchmark.
capabilities:
  - name: code-search
    kind: mcp_server
    enabled: true
    mcp_server:
      name: code-search
      transport: stdio
      command: python
      args:
        - -m
        - code_search_mcp
  - name: disabled-memory
    kind: mcp_server
    enabled: false
    mcp_server:
      name: memory
      transport: sse
      url: http://memory.example/sse
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
      NAT_PROFILE_ID: code-intelligence
""",
        encoding="utf-8",
    )
    return path


def test_load_profile_from_yaml_and_apply_enabled_capabilities(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.yml")

    applied = load_capability_profile(profile_path).apply(source_path=profile_path)

    assert applied.profile.name == "code-intelligence"
    assert [capability.name for capability in applied.enabled_capabilities] == [
        "code-search",
        "debug-skills",
        "opencode-overlay",
        "profile-env",
    ]
    assert [server.name for server in applied.mcp_servers] == ["code-search"]
    assert applied.skills_dir == "/workspace/skills/debugging"
    assert applied.env == {"NAT_PROFILE_ID": "code-intelligence"}
    assert applied.opencode_config["experimental"]["continue_loop_on_deny"] is True
    assert applied.metadata()["source_path"] == str(profile_path)


def test_load_profile_from_json(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "name": "json-profile",
            "capabilities": [{
                "name": "env",
                "kind": "env",
                "env": {
                    "NAT_PROFILE_ID": "json-profile"
                },
            }],
        }),
        encoding="utf-8",
    )

    applied = load_capability_profile(profile_path).apply()

    assert applied.profile.schema_version == "nat.capability_profile.v1"
    assert applied.env == {"NAT_PROFILE_ID": "json-profile"}


def test_missing_matching_payload_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires 'env'"):
        CapabilityProfile.model_validate({
            "name": "bad",
            "capabilities": [{
                "name": "empty-env",
                "kind": "env",
            }],
        })


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
