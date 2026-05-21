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
"""NAT config/plugin profiles for Harbor agent-harness runs.

The POC intentionally reuses NAT config vocabulary. Capabilities such as MCP
clients are declared in normal NAT sections (`function_groups`) and a named
agent harness selects them by `tool_names`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.models.task.config import MCPServerConfig
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

MCP_FUNCTION_GROUP_TYPES = {"mcp_client", "per_user_mcp_client"}


class AgentHarnessProfile(BaseModel):
    """One named agent-harness entry in a NAT-style config file."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    harness_type: str = Field(alias="_type")
    tool_names: list[str] = Field(default_factory=list)
    skills_dir: str | None = None
    opencode_config: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


class AgentHarnessRun(BaseModel):
    """One eval variant selecting a named agent-harness profile."""

    name: str
    agent_harness: str


class EvalOutputConfig(BaseModel):
    """Subset of NAT eval output config used by the POC."""

    model_config = ConfigDict(extra="allow")

    dir: str | None = None
    write_atif_workflow_output: bool = False


class EvalDatasetConfig(BaseModel):
    """Dataset pointer for the external evaluation harness."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    dataset_type: str = Field(alias="_type")
    path: str | None = None


class EvalGeneralConfig(BaseModel):
    """NAT eval-style run settings for profile comparisons."""

    model_config = ConfigDict(extra="allow")

    max_concurrency: int | None = None
    output: EvalOutputConfig | None = None
    dataset: EvalDatasetConfig | None = None
    agent_harnesses: list[AgentHarnessRun] = Field(default_factory=list)


class EvalConfig(BaseModel):
    """Subset of NAT eval config needed for profile comparison runs."""

    model_config = ConfigDict(extra="allow")

    general: EvalGeneralConfig | None = None
    evaluators: dict[str, dict[str, Any]] = Field(default_factory=dict)


class AgentHarnessRuntimeConfig(BaseModel):
    """Subset of NAT config needed to configure external agent harnesses."""

    model_config = ConfigDict(extra="allow")

    functions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    function_groups: dict[str, dict[str, Any]] = Field(default_factory=dict)
    llms: dict[str, dict[str, Any]] = Field(default_factory=dict)
    agent_harnesses: dict[str, AgentHarnessProfile] = Field(default_factory=dict)
    workflow: dict[str, Any] | None = None
    eval: EvalConfig | None = None

    def eval_agent_harness_runs(self) -> tuple[AgentHarnessRun, ...]:
        if self.eval is None or self.eval.general is None:
            return ()
        return tuple(self.eval.general.agent_harnesses)

    def apply(
        self,
        *,
        agent_harness: str | None = None,
        source_path: str | Path | None = None,
    ) -> "AppliedAgentHarnessProfile":
        if not self.agent_harnesses:
            raise ValueError("NAT config does not define any agent_harnesses")

        harness_name = agent_harness
        if harness_name is None:
            if len(self.agent_harnesses) != 1:
                names = ", ".join(sorted(self.agent_harnesses))
                raise ValueError(f"agent_harness is required; available harnesses: {names}")
            harness_name = next(iter(self.agent_harnesses))

        if harness_name not in self.agent_harnesses:
            names = ", ".join(sorted(self.agent_harnesses))
            raise ValueError(f"Unknown agent_harness {harness_name!r}; available harnesses: {names}")

        profile = self.agent_harnesses[harness_name]
        mcp_servers: list[MCPServerConfig] = []
        mcp_function_groups: list[str] = []
        mcp_tool_filters: dict[str, dict[str, list[str]]] = {}

        for tool_name in profile.tool_names:
            if tool_name in self.function_groups:
                group = self.function_groups[tool_name]
                group_type = group.get("_type")
                if group_type not in MCP_FUNCTION_GROUP_TYPES:
                    raise ValueError(
                        f"function_group {tool_name!r} has unsupported _type {group_type!r}; "
                        "only MCP function groups are supported in this POC"
                )
                mcp_servers.append(_mcp_server_from_function_group(tool_name, group))
                mcp_function_groups.append(tool_name)
                tool_filter = _mcp_tool_filter(group)
                if tool_filter:
                    mcp_tool_filters[tool_name] = tool_filter
                continue

            if tool_name in self.functions:
                raise ValueError(
                    f"tool_name {tool_name!r} is a NAT function. The current Harbor/OpenCode "
                    "POC supports MCP function_groups only."
                )

            raise ValueError(f"tool_name {tool_name!r} is not defined in functions or function_groups")

        return AppliedAgentHarnessProfile(
            config=self,
            source_path=str(source_path) if source_path else None,
            name=harness_name,
            profile=profile,
            tool_names=tuple(profile.tool_names),
            mcp_function_groups=tuple(mcp_function_groups),
            mcp_tool_filters=mcp_tool_filters,
            mcp_servers=tuple(mcp_servers),
            skills_dir=profile.skills_dir,
            opencode_config=copy.deepcopy(profile.opencode_config),
            env=dict(profile.env),
        )


@dataclass(frozen=True)
class AppliedAgentHarnessProfile:
    """Resolved agent-harness profile ready for a Harbor adapter."""

    config: AgentHarnessRuntimeConfig
    source_path: str | None
    name: str
    profile: AgentHarnessProfile
    tool_names: tuple[str, ...]
    mcp_function_groups: tuple[str, ...]
    mcp_tool_filters: dict[str, dict[str, list[str]]]
    mcp_servers: tuple[MCPServerConfig, ...]
    skills_dir: str | None
    opencode_config: dict[str, Any]
    env: dict[str, str]

    def metadata(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "agent_harness": self.name,
            "harness_type": self.profile.harness_type,
            "tool_names": list(self.tool_names),
            "mcp_function_groups": list(self.mcp_function_groups),
            "mcp_tool_filters": self.mcp_tool_filters,
        }


def _mcp_server_from_function_group(name: str, function_group: dict[str, Any]) -> MCPServerConfig:
    server = function_group.get("server")
    if not isinstance(server, dict):
        raise ValueError(f"MCP function_group {name!r} requires a 'server' mapping")

    return MCPServerConfig.model_validate({"name": name, **server})


def _mcp_tool_filter(function_group: dict[str, Any]) -> dict[str, list[str]]:
    tool_filter: dict[str, list[str]] = {}
    for key in ("include", "exclude"):
        value = function_group.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"MCP function_group {key!r} must be a list of strings")
        tool_filter[key] = list(value)
    return tool_filter


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_agent_harness_config(path: str | Path) -> AgentHarnessRuntimeConfig:
    """Load NAT-style agent-harness config from JSON or YAML."""

    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - only exercised in minimal installs
            raise RuntimeError("Loading YAML agent-harness configs requires PyYAML") from exc
        data = yaml.safe_load(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Agent-harness config must be a mapping: {config_path}")

    return AgentHarnessRuntimeConfig.model_validate(data)
