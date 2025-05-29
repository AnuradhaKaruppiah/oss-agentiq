# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal

from pydantic import Field
from pydantic import HttpUrl

from aiq.cli.register_workflow import register_server
from aiq.data_models.server import ServerBaseConfig


class MCPServerConfig(ServerBaseConfig, name="mcp_server"):
    """
    Configuration for MCP server instances.
    """
    transport_type: Literal["sse", "stdio", "streamable-http"] = Field(default="sse",
                                                                       description="The type of transport to use")
    url: HttpUrl | None = Field(default=None, description="The URL of the server (for SSE/StreamableHTTP mode)")
    command: str | None = Field(default=None,
                                description="The command to run for stdio mode (e.g. 'docker' or 'python')")
    args: list[str] | None = Field(default=None, description="Additional arguments for the stdio command")
    env: dict[str, str] | None = Field(default=None, description="Environment variables to set for the stdio process")

    def model_post_init(self, __context):
        """Validate configuration based on transport type."""
        super().model_post_init(__context)

        if self.transport_type == 'stdio':
            if self.url is not None:
                raise ValueError("url should not be set when using stdio transport type")
            if not self.command:
                raise ValueError("command is required when using stdio transport type")
        elif self.transport_type in ['streamable-http', 'sse']:
            if self.command is not None or self.args is not None or self.env is not None:
                raise ValueError(
                    "command, args, and env should not be set when using sse/streamable-http transport type")
            if not self.url:
                raise ValueError("url is required when using sse/streamable-http transport type")


@register_server(config_type=MCPServerConfig)
def mcp_server(config: MCPServerConfig):
    """
    Create an MCP server instance.
    """
    from aiq.tool.mcp.mcp_server import MCPServer
    return MCPServer(config)
