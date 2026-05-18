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
"""Runtime capability profiles for Harbor agent runs.

Profiles describe runtime agent capabilities that should vary independently of a
benchmark/task definition: MCP servers, skills directories, harness-specific
config, and environment variables.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from harbor.models.task.config import MCPServerConfig
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

CapabilityKind = Literal["mcp_server", "skills_dir", "opencode_config", "env"]


class CapabilityEntry(BaseModel):
    """One optional runtime capability in a profile."""

    name: str
    kind: CapabilityKind
    enabled: bool = True
    description: str | None = None
    mcp_server: MCPServerConfig | None = None
    skills_dir: str | None = None
    opencode_config: dict[str, Any] | None = None
    env: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_matching_payload(self) -> "CapabilityEntry":
        payload = getattr(self, self.kind)
        if payload in (None, {}, ""):
            raise ValueError(f"Capability {self.name!r} of kind {self.kind!r} requires {self.kind!r}")
        return self


class CapabilityProfile(BaseModel):
    """Config-driven description of runtime agent capabilities."""

    schema_version: str = "nat.capability_profile.v1"
    name: str
    description: str | None = None
    capabilities: list[CapabilityEntry] = Field(default_factory=list)

    def apply(self, *, source_path: str | Path | None = None) -> "AppliedCapabilityProfile":
        return AppliedCapabilityProfile.from_profile(self, source_path=source_path)


@dataclass(frozen=True)
class AppliedCapabilityProfile:
    """Enabled capability payloads prepared for a specific harness adapter."""

    profile: CapabilityProfile
    source_path: str | None
    enabled_capabilities: tuple[CapabilityEntry, ...]
    mcp_servers: tuple[MCPServerConfig, ...]
    skills_dir: str | None
    opencode_config: dict[str, Any]
    env: dict[str, str]

    @classmethod
    def from_profile(
        cls,
        profile: CapabilityProfile,
        *,
        source_path: str | Path | None = None,
    ) -> "AppliedCapabilityProfile":
        enabled = tuple(capability for capability in profile.capabilities if capability.enabled)
        mcp_servers: list[MCPServerConfig] = []
        skills_dir: str | None = None
        opencode_config: dict[str, Any] = {}
        env: dict[str, str] = {}

        for capability in enabled:
            if capability.kind == "mcp_server":
                assert capability.mcp_server is not None
                mcp_servers.append(capability.mcp_server)
            elif capability.kind == "skills_dir":
                skills_dir = capability.skills_dir
            elif capability.kind == "opencode_config":
                assert capability.opencode_config is not None
                opencode_config = deep_merge_dicts(opencode_config, capability.opencode_config)
            elif capability.kind == "env":
                assert capability.env is not None
                env.update(capability.env)

        return cls(
            profile=profile,
            source_path=str(source_path) if source_path else None,
            enabled_capabilities=enabled,
            mcp_servers=tuple(mcp_servers),
            skills_dir=skills_dir,
            opencode_config=opencode_config,
            env=env,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.profile.schema_version,
            "name": self.profile.name,
            "description": self.profile.description,
            "source_path": self.source_path,
            "enabled_capabilities": [
                {
                    "name": capability.name,
                    "kind": capability.kind,
                }
                for capability in self.enabled_capabilities
            ],
        }


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_capability_profile(path: str | Path) -> CapabilityProfile:
    """Load a capability profile from JSON or YAML."""

    profile_path = Path(path)
    raw = profile_path.read_text(encoding="utf-8")
    if profile_path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - only exercised in minimal installs
            raise RuntimeError("Loading YAML capability profiles requires PyYAML") from exc
        data = yaml.safe_load(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Capability profile must be a mapping: {profile_path}")

    return CapabilityProfile.model_validate(data)
