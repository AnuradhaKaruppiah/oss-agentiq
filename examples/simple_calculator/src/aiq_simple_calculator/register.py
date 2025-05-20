# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import typing
import uuid

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.context import AIQContext
from aiq.builder.function_info import FunctionInfo
from aiq.builder.intermediate_step_manager import IntermediateStepManager
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData

logger = logging.getLogger(__name__)


def validate_number_count(numbers: list[str], expected_count: int, action: str) -> str | None:
    if len(numbers) < expected_count:
        return f"Provide at least {expected_count} numbers to {action}."
    if len(numbers) > expected_count:
        return f"This tool only supports {action} between {expected_count} numbers."
    return None


class InequalityToolConfig(FunctionBaseConfig, name="calculator_inequality"):
    pass


@register_function(config_type=InequalityToolConfig)
async def calculator_inequality(tool_config: InequalityToolConfig, builder: Builder):

    import re

    async def _calculator_inequality(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        validation_error = validate_number_count(numbers, expected_count=2, action="compare")
        if validation_error:
            return validation_error
        a = int(numbers[0])
        b = int(numbers[1])
        if a > b:
            return f"First number {a} is greater than the second number {b}"
        if a < b:
            return f"First number {a} is less than the second number {b}"

        return f"First number {a} is equal to the second number {b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_inequality,
        description=("This is a mathematical tool used to perform an inequality comparison between two numbers. "
                     "It takes two numbers as an input and determines if one is greater or are equal."))


class MultiplyToolConfig(FunctionBaseConfig, name="calculator_multiply"):
    pass


@register_function(config_type=MultiplyToolConfig)
async def calculator_multiply(config: MultiplyToolConfig, builder: Builder):

    import re

    async def _calculator_multiply(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        validation_error = validate_number_count(numbers, expected_count=2, action="multiply")
        if validation_error:
            return validation_error
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The product of {a} * {b} is {a * b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_multiply,
        description=("This is a mathematical tool used to multiply two numbers together. "
                     "It takes 2 numbers as an input and computes their numeric product as the output."))


class DivisionToolConfig(FunctionBaseConfig, name="calculator_divide"):
    pass


@register_function(config_type=DivisionToolConfig)
async def calculator_divide(config: DivisionToolConfig, builder: Builder):

    import re

    async def _calculator_divide(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        validation_error = validate_number_count(numbers, expected_count=2, action="divide")
        if validation_error:
            return validation_error
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The result of {a} / {b} is {a / b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_divide,
        description=("This is a mathematical tool used to divide one number by another. "
                     "It takes 2 numbers as an input and computes their numeric quotient as the output."))


class SubtractToolConfig(FunctionBaseConfig, name="calculator_subtract"):
    pass


@register_function(config_type=SubtractToolConfig)
async def calculator_subtract(config: SubtractToolConfig, builder: Builder):

    import re

    async def _calculator_subtract(text: str) -> str:
        numbers = re.findall(r"\d+", text)
        validation_error = validate_number_count(numbers, expected_count=2, action="subtract")
        if validation_error:
            return validation_error
        a = int(numbers[0])
        b = int(numbers[1])

        return f"The result of {a} - {b} is {a - b}"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _calculator_subtract,
        description=("This is a mathematical tool used to subtract one number from another. "
                     "It takes 2 numbers as an input and computes their numeric difference as the output."))


class SimpleCalculatorConfig(FunctionBaseConfig, name="simple_calculator"):
    pass


class CustomIntermediateStepPayload(IntermediateStepPayload):
    UUID: str = Field("", description="UUID")
    name: str = Field("", description="Tool name")
    data: StreamEventData = Field("", description="Data")
    event_type: IntermediateStepType = IntermediateStepType.CUSTOM_START


class CustomIntermediateStepPayloadEnd(IntermediateStepPayload):
    UUID: str = Field("", description="UUID")
    name: str = Field("", description="Tool name")
    data: StreamEventData = Field("", description="Data")
    event_type: IntermediateStepType = IntermediateStepType.CUSTOM_END


def custom_start_tool(step_manager: IntermediateStepManager,
                      name: str,
                      input_data: typing.Any | None = None,
                      output_data: typing.Any | None = None):
    uid = str(uuid.uuid4())
    start_step = CustomIntermediateStepPayload(UUID=uid,
                                               name=name,
                                               data=StreamEventData(input=input_data, output=output_data),
                                               event_type=IntermediateStepType.CUSTOM_START)
    step_manager.push_intermediate_step(start_step)

    return uid


def custom_end_tool(step_manager: IntermediateStepManager,
                    uid: str,
                    name: str,
                    input_data: typing.Any | None = None,
                    output_data: typing.Any | None = None):

    end_step = CustomIntermediateStepPayloadEnd(UUID=uid,
                                                name=name,
                                                data=StreamEventData(input=input_data, output=output_data),
                                                event_type=IntermediateStepType.CUSTOM_END)
    step_manager.push_intermediate_step(end_step)


@register_function(config_type=SimpleCalculatorConfig)
async def simple_calculator(config: SimpleCalculatorConfig, builder: Builder):
    import re

    # Get all calculator tools
    multiply_tool = builder.get_function(name="calculator_multiply")
    divide_tool = builder.get_function(name="calculator_divide")
    subtract_tool = builder.get_function(name="calculator_subtract")
    inequality_tool = builder.get_function(name="calculator_inequality")

    # Get datetime tool
    datetime_tool = builder.get_function(name="current_datetime")

    enable_custom_steps = True
    # Get step manager
    if enable_custom_steps:
        step_manager = AIQContext.get().intermediate_step_manager
    else:
        step_manager = None

    async def _simple_calculator(input_message: str) -> str:
        # Use custom steps if step_manager is available
        if step_manager is not None:
            uid = custom_start_tool(step_manager=step_manager, name="datetime", input_data=input_message)

        # First get current datetime with custom tools
        current_time = await datetime_tool.ainvoke(input_message)

        if step_manager is not None:
            custom_end_tool(step_manager=step_manager, uid=uid, name="datetime", output_data=current_time)

        response = f"Current time: {current_time}\n\n"

        # Extract numbers from input
        numbers = re.findall(r"\d+", input_message)
        if len(numbers) < 2:
            return f"{response}Error: Please provide at least 2 numbers for calculation."

        # Call all calculator tools
        multiply_result = await multiply_tool.ainvoke(input_message)
        divide_result = await divide_tool.ainvoke(input_message)
        subtract_result = await subtract_tool.ainvoke(input_message)
        inequality_result = await inequality_tool.ainvoke(input_message)

        # Combine all results
        response += "Calculation Results:\n"
        response += f"{multiply_result}\n"
        response += f"{divide_result}\n"
        response += f"{subtract_result}\n"
        response += f"{inequality_result}\n"

        return response

    yield FunctionInfo.from_fn(
        _simple_calculator,
        description=("A calculator that performs multiple mathematical operations on two numbers. "
                     "It first gets the current time and then performs multiplication, division, "
                     "subtraction, and inequality comparison on the provided numbers."))
