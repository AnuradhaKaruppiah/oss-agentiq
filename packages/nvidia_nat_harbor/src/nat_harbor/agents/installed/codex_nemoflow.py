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
"""Experimental Harbor wrapper for Codex with NeMo-Flow gateway enabled."""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


class CodexNeMoFlow(Codex):
    """Run Codex through the NeMo-Flow coding-agent gateway inside Harbor."""

    _DEFAULT_NEMO_FLOW_REPO = "external/nemo-flow"
    _DEFAULT_HOST_NEMO_FLOW_BIN_CANDIDATES = (
        "target/x86_64-unknown-linux-gnu/release/nemo-flow",
        "target/release/nemo-flow",
    )
    _CONTAINER_NEMO_FLOW_BIN = PurePosixPath("/usr/local/bin/nemo-flow")
    _CONTAINER_ATIF_DIR = PurePosixPath(EnvironmentPaths.agent_dir / "nemo-flow-atif")
    _CANONICAL_ATIF_FILENAME = "trajectory.json"
    _NVIDIA_PROVIDER = "nvidia"
    _NVIDIA_MODEL_ALIASES = {
        "opus-frontier": "aws/anthropic/claude-opus-4-5",
    }

    def __init__(
        self,
        *args: Any,
        nemo_flow_repo: str | None = None,
        nemo_flow_bin: str | None = None,
        fail_missing_nemoflow_atif: bool = True,
        fail_missing_native_codex_atif: bool = False,
        codex_model: str | None = None,
        codex_enable_unified_exec: bool = True,
        nemo_flow_debug_gateway: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._nemo_flow_repo = Path(
            nemo_flow_repo
            or os.environ.get("NEMO_FLOW_REPO", self._DEFAULT_NEMO_FLOW_REPO)
        ).resolve()
        self._nemo_flow_bin = self._resolve_host_nemo_flow_bin(nemo_flow_bin)
        self._fail_missing_nemoflow_atif = fail_missing_nemoflow_atif
        self._fail_missing_native_codex_atif = fail_missing_native_codex_atif
        self._codex_model_override = codex_model
        self._codex_enable_unified_exec = codex_enable_unified_exec
        self._nemo_flow_debug_gateway = nemo_flow_debug_gateway

    @staticmethod
    def name() -> str:
        return "codex-nemoflow"

    def _resolve_host_nemo_flow_bin(self, explicit: str | None) -> Path:
        value = explicit or os.environ.get("NEMO_FLOW_BIN")
        if value:
            path = Path(value).expanduser().resolve()
            candidates = [path]
        else:
            candidates = [
                self._nemo_flow_repo / relative
                for relative in self._DEFAULT_HOST_NEMO_FLOW_BIN_CANDIDATES
            ]
            path = candidates[0]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
        if not path.is_file():
            raise FileNotFoundError(
                "NeMo-Flow gateway binary is not ready. Expected executable at "
                f"one of: {', '.join(str(candidate) for candidate in candidates)}. "
                "Build a portable binary with "
                "`RUSTFLAGS='-C target-feature=+crt-static' cargo build --release "
                "-p nemo-flow-cli --target x86_64-unknown-linux-gnu`, "
                "or pass --ak nemo_flow_bin=<path>."
            )
        return path

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        await environment.upload_file(self._nemo_flow_bin, "/tmp/nemo-flow")
        await self.exec_as_root(
            environment,
            command=(
                f"install -m 0755 /tmp/nemo-flow {self._CONTAINER_NEMO_FLOW_BIN.as_posix()} && "
                f"{self._CONTAINER_NEMO_FLOW_BIN.as_posix()} --help >/dev/null"
            ),
        )

    def _provider_env_and_gateway_base_url(self) -> tuple[dict[str, str], str | None]:
        provider = (
            self.model_name.split("/", 1)[0]
            if self.model_name and "/" in self.model_name
            else "openai"
        )
        if provider == self._NVIDIA_PROVIDER:
            api_key = self._get_env("NVIDIA_API_KEY")
            base_url = self._get_env("NVIDIA_BASE_URL")
            if not api_key:
                raise ValueError(
                    "Provider 'nvidia' requires NVIDIA_API_KEY in the host environment or extra_env"
                )
            if not base_url:
                raise ValueError(
                    "Provider 'nvidia' requires NVIDIA_BASE_URL in the host environment or extra_env"
                )
            return {
                "NVIDIA_API_KEY": api_key,
                "NVIDIA_BASE_URL": base_url,
                # Codex's OpenAI-compatible auth path is preserved while the
                # gateway controls the upstream base URL.
                "OPENAI_API_KEY": api_key,
            }, base_url

        api_key = self._get_env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"Provider {provider!r} requires OPENAI_API_KEY in the host environment or extra_env"
            )
        env = {"OPENAI_API_KEY": api_key}
        base_url = self._get_env("OPENAI_BASE_URL")
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        return env, base_url

    def _codex_model(self) -> str:
        if self._codex_model_override:
            return self._codex_model_override
        if not self.model_name:
            raise ValueError("Model name is required")
        if "/" not in self.model_name:
            return self.model_name
        provider, model = self.model_name.split("/", 1)
        if provider == self._NVIDIA_PROVIDER:
            return self._NVIDIA_MODEL_ALIASES.get(model, model)
        return model

    def _build_auth_setup_command(self, env: dict[str, str], remote_auth_path: str) -> str:
        auth_json_path = self._resolve_auth_json_path()
        if auth_json_path:
            return f'ln -sf {shlex.quote(remote_auth_path)} "$CODEX_HOME/auth.json"\n'

        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", self._get_env("OPENAI_API_KEY") or "")
        return (
            f"cat >{shlex.quote(remote_auth_path)} <<EOF\n"
            '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF\n'
            f"ln -sf {shlex.quote(remote_auth_path)} "
            '"$CODEX_HOME/auth.json"\n'
        )

    async def _upload_auth_json_if_needed(
        self,
        environment: BaseEnvironment,
        remote_auth_path: str,
    ) -> None:
        auth_json_path = self._resolve_auth_json_path()
        if not auth_json_path:
            return
        await environment.upload_file(auth_json_path, remote_auth_path)
        if environment.default_user is not None:
            await self.exec_as_root(
                environment,
                command=f"chown {environment.default_user} {shlex.quote(remote_auth_path)}",
            )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)
        model = self._codex_model()
        provider_env, gateway_base_url = self._provider_env_and_gateway_base_url()

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""
        unified_exec_arg = (
            "--enable unified_exec " if self._codex_enable_unified_exec else ""
        )

        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        remote_secrets_dir = self._REMOTE_CODEX_SECRETS_DIR.as_posix()
        remote_auth_path = (self._REMOTE_CODEX_SECRETS_DIR / "auth.json").as_posix()

        env: dict[str, str] = {
            "CODEX_HOME": remote_codex_home,
            **provider_env,
        }
        if self._nemo_flow_debug_gateway:
            env["NEMO_FLOW_DEBUG_GATEWAY_DIR"] = (
                self._CONTAINER_ATIF_DIR / "debug"
            ).as_posix()

        await self.exec_as_agent(
            environment,
            command=(
                f'mkdir -p "$CODEX_HOME" {shlex.quote(remote_secrets_dir)} '
                f"{shlex.quote(EnvironmentPaths.agent_dir.as_posix())} "
                f"{shlex.quote(self._CONTAINER_ATIF_DIR.as_posix())}"
            ),
            env=env,
        )

        await self._upload_auth_json_if_needed(environment, remote_auth_path)
        setup_command = self._build_auth_setup_command(env, remote_auth_path)

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            setup_command += f"\n{mcp_command}"

        if setup_command.strip():
            await self.exec_as_agent(environment, command=setup_command, env=env)

        gateway_args = [
            self._CONTAINER_NEMO_FLOW_BIN.as_posix(),
            "run",
            "--agent",
            "codex",
            "--atif-dir",
            self._CONTAINER_ATIF_DIR.as_posix(),
        ]
        if gateway_base_url:
            gateway_args.extend(["--openai-base-url", gateway_base_url])

        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                    f"{shlex.join(gateway_args)} -- "
                    "codex exec "
                    "--dangerously-bypass-approvals-and-sandbox "
                    "--skip-git-repo-check "
                    f"--model {shlex.quote(model)} "
                    "--json "
                    f"{unified_exec_arg}"
                    f"{cli_flags_arg}"
                    "-- "
                    f"{escaped_instruction} "
                    f"2>&1 </dev/null | tee {EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME}"
                ),
                env=env,
            )
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()}\n"
                        'if [ -d "$CODEX_HOME/sessions" ]; then\n'
                        f"  rm -rf {(EnvironmentPaths.agent_dir / 'sessions').as_posix()}\n"
                        f'  cp -R "$CODEX_HOME/sessions" '
                        f'{(EnvironmentPaths.agent_dir / "sessions").as_posix()}\n'
                        "fi"
                    ),
                    env=env,
                )
            except Exception:
                pass
            try:
                await self.exec_as_agent(
                    environment,
                    command=f'rm -rf {shlex.quote(remote_secrets_dir)} "$CODEX_HOME"',
                    env=env,
                )
            except Exception:
                pass

    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)

        native_atif_path = self.logs_dir / "trajectory.json"
        atif_dir = self.logs_dir / "nemo-flow-atif"
        atif_files = sorted(atif_dir.glob("*.atif.json")) if atif_dir.exists() else []
        canonical_atif_path = atif_dir / self._CANONICAL_ATIF_FILENAME

        if self._fail_missing_native_codex_atif and not native_atif_path.exists():
            raise FileNotFoundError(
                f"Missing native Codex ATIF artifact: {native_atif_path}"
            )
        if (
            self._fail_missing_nemoflow_atif
            and not atif_files
            and not canonical_atif_path.exists()
        ):
            raise FileNotFoundError(
                f"Missing NeMo-Flow Codex ATIF artifact under: {atif_dir}"
            )

        if atif_files:
            newest_atif = max(atif_files, key=lambda path: path.stat().st_mtime_ns)
            if newest_atif != canonical_atif_path:
                canonical_atif_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(newest_atif, canonical_atif_path)

        metadata = dict(context.metadata or {})
        metadata["nemo_flow_native_atif_path"] = str(native_atif_path)
        metadata["nemo_flow_native_atif_exists"] = native_atif_path.exists()
        metadata["nemo_flow_gateway_atif_dir"] = str(atif_dir)
        metadata["nemo_flow_gateway_atif_paths"] = [str(path) for path in atif_files]
        metadata["nemo_flow_gateway_canonical_atif_path"] = str(canonical_atif_path)
        metadata["nemo_flow_gateway_canonical_atif_exists"] = canonical_atif_path.exists()
        context.metadata = metadata
