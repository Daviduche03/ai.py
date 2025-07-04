import json
import random
from typing import AsyncGenerator, Dict, Any, List
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from google import genai
from google.genai import types
from ai.providers.base import BaseProvider
from ai.tools import Tool
from pydantic import BaseModel
import logging
import inspect

logger = logging.getLogger(__name__)


class StreamEvent(BaseModel):
    event: str
    data: Any


class GoogleProvider(BaseProvider):
    def __init__(self, client: genai.Client):
        self.client = client

    async def stream(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        tools: list[Dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamEvent, None]:
        contents = []
        system_instruction = ""

        for msg in messages:
            if msg is None:
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                # Extract system instruction separately
                if role == "system":
                    system_instruction = content
                    continue
                # Map roles to Google's expected format
                elif role == "assistant":
                    role = "model"
                elif role == "tool":
                    # Handle tool messages by converting them to model responses
                    tool_call_id = msg.get("tool_call_id")
                    tool_content = msg.get("content")
                    if tool_content:
                        # Add tool result as a model message
                        contents.append(
                            {
                                "role": "model",
                                "parts": [{"text": f"Tool result: {tool_content}"}],
                            }
                        )
                    continue
                elif role not in ["user", "model"]:
                    # Default unknown roles to user
                    role = "user"
                contents.append({"role": role, "parts": [{"text": content}]})

        gemini_tools = types.Tool(function_declarations=tools)  # ✅ Wrap tools properly

        if not system_instruction:
            raise ValueError("System instruction must be provided.")

        # Ensure we have at least some content
        if not contents:
            raise ValueError("At least one user or assistant message is required.")

        try:
            stream = self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[gemini_tools],
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield StreamEvent(event="text", data=chunk.text)
                if chunk.function_calls:
                    yield StreamEvent(event="tool_calls", data=chunk.function_calls)

        except Exception as e:
            print(f"Error generating content with Google AI: {e}")
            raise

    async def generate(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        tools: list[Dict[str, Any]] | None = None,
        **kwargs,
    ) -> ChatCompletion:
        contents = []
        system_instruction = ""

        for msg in messages:
            if msg is None:
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                # Extract system instruction separately
                if role == "system":
                    system_instruction = content
                    continue
                # Map roles to Google's expected format
                elif role == "assistant":
                    role = "model"
                elif role == "tool":
                    # Handle tool messages by converting them to model responses
                    tool_call_id = msg.get("tool_call_id")
                    tool_content = msg.get("content")
                    if tool_content:
                        # Add tool result as a model message
                        contents.append(
                            {
                                "role": "model",
                                "parts": [{"text": f"Tool result: {tool_content}"}],
                            }
                        )
                    continue
                elif role not in ["user", "model"]:
                    # Default unknown roles to user
                    role = "user"
                contents.append({"role": role, "parts": [{"text": content}]})

        gemini_tools = types.Tool(function_declarations=tools)  # ✅ Wrap tools properly

        if not system_instruction:
            raise ValueError("System instruction must be provided.")

        # Ensure we have at least some content
        if not contents:
            raise ValueError("At least one user or assistant message is required.")
        try:
            completion = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[gemini_tools],
                ),
            )
            return completion
        except Exception as e:
            print(f"Error generating content with Google AI: {e}")
            raise

    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        if not tools:
            return None
        return [tool.as_google_tool() for tool in tools]

    async def process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_map: Dict[str, Tool],
    ) -> List[Dict[str, Any]]:
        print("tool_calls", tool_calls)
        if tool_calls:
            tool_results = []
            for tool_call in tool_calls:
                print("individual calls", tool_call, "name: ", tool_call.name)
                tool_name = tool_call.name
                tool_args = tool_call.args
                tool = tool_map.get(tool_name)
                if tool:
                    if inspect.iscoroutinefunction(tool.execute):
                        result = await tool.execute(tool.parameters(**tool_args))
                    else:
                        result = tool.execute(tool.parameters(**tool_args))

                    logger.info(
                        f"Tool '{tool_name}' called with args {tool_args}. Result: {result}"
                    )
                    tool_results.append(
                        {
                            "tool_call_id": random.randint(
                                1000, 9999
                            ),  # Simulating a tool call ID
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(result),
                        }
                    )

            return tool_results
