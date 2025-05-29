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

from aiq.data_models.server import ServerBaseConfig
from aiq.registry import register_server


class A2AServerConfig(ServerBaseConfig, name="a2a_server"):
    """
    Configuration for A2A server instances.
    """
    server_type: Literal["a2a"] = "a2a"
    url: HttpUrl = Field(description="The URL of the A2A server")


@register_server(config_type=A2AServerConfig)
def a2a_server(config: A2AServerConfig):
    """
    Create an A2A server instance.
    """
    from aiq.tool.a2a.a2a_server import A2AServer
    return A2AServer(config)
