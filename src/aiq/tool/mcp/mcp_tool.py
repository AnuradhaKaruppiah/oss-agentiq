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

import logging
from typing import Literal

from pydantic import Field
from pydantic import HttpUrl

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import ServerRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class MCPToolConfig(FunctionBaseConfig, name="mcp_tool_wrapper"):
    """
    Function which connects to a Model Context Protocol (MCP) server and wraps the selected tool as an AIQ Toolkit
    function.
    """
    # Server reference for indirect configuration
    server: ServerRef | None = Field(default=None, description="Reference to the MCP server configuration")

    # Direct configuration fields (for backward compatibility with SSE/streamable-http)
    url: HttpUrl | None = Field(default=None, description="The URL of the MCP server (for SSE/streamable-http mode)")
    transport_type: Literal["sse", "stdio", "streamable-http"] = Field(default="sse",
                                                                       description="The type of transport to use")
    command: str | None = Field(default=None,
                                description="The command to run for stdio mode (e.g. 'docker' or 'python')")
    args: list[str] | None = Field(default=None, description="Additional arguments for the stdio command")
    env: dict[str, str] | None = Field(default=None, description="Environment variables to set for the stdio process")

    # Common fields
    mcp_tool_name: str = Field(description="The name of the tool served by the MCP Server that you want to use")
    description: str | None = Field(default=None,
                                    description="""
        Description for the tool that will override the description provided by the MCP server. Should only be used if
        the description provided by the server is poor or nonexistent
        """)
    return_exception: bool = Field(default=True,
                                   description="""
        If true, the tool will return the exception message if the tool call fails.
        If false, raise the exception.
        """)

    def model_post_init(self, __context):
        """Validate configuration based on transport type and configuration method."""
        super().model_post_init(__context)

        # Check if using direct or indirect configuration
        using_server_ref = self.server is not None
        using_direct_config = self.url is not None or self.command is not None

        if not using_server_ref and not using_direct_config:
            raise ValueError("Either server reference or direct configuration must be provided")

        if using_server_ref and using_direct_config:
            raise ValueError("Cannot use both server reference and direct configuration")

        if using_server_ref:
            # When using server reference, no direct configuration should be provided
            if self.url is not None or self.command is not None or self.args is not None or self.env is not None:
                raise ValueError("Direct configuration fields should not be set when using server reference")
        else:
            # Direct configuration validation
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


@register_function(config_type=MCPToolConfig)
async def mcp_tool(config: MCPToolConfig, builder: Builder):
    """
    Generate an AIQ Toolkit Function that wraps a tool provided by the MCP server.
    """
    from aiq.tool.mcp.mcp_client import MCPSSEClient
    from aiq.tool.mcp.mcp_client import MCPStdioClient
    from aiq.tool.mcp.mcp_client import MCPStreamableHTTPClient
    from aiq.tool.mcp.mcp_client import MCPToolClient

    # Get server configuration either from reference or direct config
    if config.server is not None:
        server_config = builder.get_server(config.server)
        if server_config.server_type != "mcp":
            raise ValueError(f"Server {config.server} is not an MCP server")

        transport_type = server_config.transport_type
        url = server_config.url
        command = server_config.command
        args = server_config.args
        env = server_config.env
    else:
        transport_type = config.transport_type
        url = config.url
        command = config.command
        args = config.args
        env = config.env

    # Initialize the client
    if transport_type == 'stdio':
        source = f"{command} {' '.join(args) if args else ''}"
        client = MCPStdioClient(command=command, args=args, env=env)
    elif transport_type == 'streamable-http':
        source = str(url)
        client = MCPStreamableHTTPClient(url=source)
    elif transport_type == 'sse':
        source = str(url)
        client = MCPSSEClient(url=source)
    else:
        raise ValueError(f"Invalid transport type: {transport_type}")

    async with client:
        # If the tool is found create a MCPToolClient object and set the description if provided
        tool: MCPToolClient = await client.get_tool(config.mcp_tool_name)
        if config.description:
            tool.set_description(description=config.description)

        logger.info("Configured to use tool: %s from MCP server at %s", tool.name, source)

        def _convert_from_str(input_str: str) -> tool.input_schema:
            return tool.input_schema.model_validate_json(input_str)

        async def _response_fn(tool_input: tool.input_schema | None = None, **kwargs) -> str:
            # Run the tool, catching any errors and sending to agent for correction
            try:
                if tool_input:
                    args = tool_input.model_dump()
                    return await tool.acall(args)

                _ = tool.input_schema.model_validate(kwargs)
                return await tool.acall(kwargs)
            except Exception as e:
                if config.return_exception:
                    if tool_input:
                        logger.warning("Error calling tool %s with serialized input: %s",
                                       tool.name,
                                       tool_input.model_dump(),
                                       exc_info=True)
                    else:
                        logger.warning("Error calling tool %s with input: %s", tool.name, kwargs, exc_info=True)
                    return str(e)
                # If the tool call fails, raise the exception.
                raise

        yield FunctionInfo.create(single_fn=_response_fn,
                                  description=tool.description,
                                  input_schema=tool.input_schema,
                                  converters=[_convert_from_str])
