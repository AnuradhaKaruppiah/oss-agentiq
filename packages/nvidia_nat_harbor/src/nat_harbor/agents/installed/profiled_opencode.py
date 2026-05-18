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
"""OpenCode wrapper that applies NAT runtime capability profiles."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from harbor.agents.installed.opencode import OpenCode
from harbor.models.agent.context import AgentContext

from nat_harbor.capability_profiles import AppliedCapabilityProfile
from nat_harbor.capability_profiles import deep_merge_dicts
from nat_harbor.capability_profiles import load_capability_profile


class ProfiledOpenCode(OpenCode):
    """Run Harbor OpenCode with a NAT-owned runtime capability profile.

    The benchmark, task, and environment remain unchanged. The profile only
    overlays the agent's runtime capability surface.
    """

    def __init__(
        self,
        *args: Any,
        capability_profile: str | Path | None = None,
        opencode_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        profile_application = self._load_capability_profile(capability_profile)
        if profile_application:
            self._apply_capability_profile_kwargs(kwargs, profile_application)

        profile_config = copy.deepcopy(profile_application.opencode_config) if profile_application else {}
        merged_config = deep_merge_dicts(profile_config, opencode_config or {})
        super().__init__(*args, opencode_config=merged_config, **kwargs)
        self._capability_profile_application = profile_application

    @staticmethod
    def name() -> str:
        return "profiled-opencode"

    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        self._record_capability_profile(context)

    def _load_capability_profile(self, capability_profile: str | Path | None) -> AppliedCapabilityProfile | None:
        profile_path = capability_profile or os.environ.get("NAT_HARBOR_CAPABILITY_PROFILE")
        if not profile_path:
            return None
        return load_capability_profile(profile_path).apply(source_path=profile_path)

    @staticmethod
    def _apply_capability_profile_kwargs(
        kwargs: dict[str, Any],
        profile_application: AppliedCapabilityProfile,
    ) -> None:
        if profile_application.mcp_servers:
            kwargs["mcp_servers"] = [*(kwargs.get("mcp_servers") or []), *profile_application.mcp_servers]

        if profile_application.skills_dir and not kwargs.get("skills_dir"):
            kwargs["skills_dir"] = profile_application.skills_dir

        if profile_application.env:
            extra_env = dict(profile_application.env)
            extra_env.update(kwargs.get("extra_env") or {})
            kwargs["extra_env"] = extra_env

    def _record_capability_profile(self, context: AgentContext) -> None:
        if not self._capability_profile_application:
            return

        metadata = dict(context.metadata or {})
        metadata["nat_capability_profile"] = self._capability_profile_application.metadata()
        context.metadata = metadata
