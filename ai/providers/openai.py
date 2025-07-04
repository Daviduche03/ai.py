import json
from typing import AsyncGenerator, Dict, Any, List
import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from ai.providers.base import BaseProvider
from ai.tools import Tool
from pydantic import BaseModel
import logging
import inspect

logger = logging.getLogger(__name__)


class StreamEvent(BaseModel):
    event: str
    data: Any


class OpenAIProvider(BaseProvider):
    def __init__(self, client: openai.AsyncOpenAI):
        self.client = client

    async def stream(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        tools: list[Dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamEvent, None]:
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            tools=tools,
            **kwargs,
        )

        tool_call_ids: Dict[int, str] = {}
        tool_call_names: Dict[int, str] = {}
        tool_call_args: Dict[int, str] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield StreamEvent(event="text", data=delta.content)

            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    index = tool_call_chunk.index
                    if tool_call_chunk.id:
                        tool_call_ids[index] = tool_call_chunk.id
                    if tool_call_chunk.function:
                        if tool_call_chunk.function.name:
                            tool_call_names[index] = tool_call_chunk.function.name
                        if tool_call_chunk.function.arguments:
                            if index not in tool_call_args:
                                tool_call_args[index] = ""
                            tool_call_args[index] += tool_call_chunk.function.arguments

        if tool_call_ids:
            formatted_tool_calls = []
            for index, tool_id in tool_call_ids.items():
                formatted_tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_call_names.get(index),
                            "arguments": tool_call_args.get(index, ""),
                        },
                    }
                )
            yield StreamEvent(event="tool_calls", data=formatted_tool_calls)

    async def generate(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        tools: list[Dict[str, Any]] | None = None,
        **kwargs,
    ) -> ChatCompletion:
        completion = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            tools=tools,
            **kwargs,
        )
        return completion

    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [tool.as_openai_tool() for tool in tools]

    async def process_tool_calls(
        self,
        tool_calls: List[
            Any
        ],  # Changed from Dict to Any to handle ChatCompletionMessageToolCall objects
        tool_map: Dict[str, Tool],
    ) -> List[Dict[str, Any]]:
        print("tool_calls", tool_calls)

        tool_results = []

        if tool_calls:
            for tool_call in tool_calls:
                # Handle both dict and ChatCompletionMessageToolCall object formats
                if hasattr(
                    tool_call, "function"
                ):  # ChatCompletionMessageToolCall object
                    print(
                        "individual calls: ",
                        tool_call,
                        "function: ",
                        tool_call.function.name,
                    )
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id
                else:  # Dict format (fallback)
                    print(
                        "individual calls: ",
                        tool_call,
                        "function: ",
                        tool_call["function"]["name"],
                    )
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    tool_call_id = tool_call["id"]

                tool = tool_map.get(tool_name)

                if tool:
                    if inspect.iscoroutinefunction(tool.execute):
                        result = await tool.execute(tool.parameters(**tool_args))
                    else:
                        result = tool.execute(tool.parameters(**tool_args))

                    logger.info(
                        f"Tool '{tool_name}' executed with args {tool_args}, result: {result}"
                    )

                    tool_results.append(
                        {
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(result),
                        }
                    )
                else:
                    logger.warning(f"Tool '{tool_name}' not found in tool_map")

        return tool_results
