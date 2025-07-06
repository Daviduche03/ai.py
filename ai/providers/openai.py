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
    """
    Represents a streaming event from the AI provider.
    
    This class encapsulates streaming data chunks that can contain
    either text content or tool calls.
    
    Attributes:
        event (str): The type of event ('text' or 'tool_calls')
        data (Any): The event data (text string or tool call objects)
    """
    event: str
    data: Any


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider implementation for the AI SDK.
    
    This class handles communication with OpenAI's API, including streaming
    responses, generating completions, and processing tool calls.
    
    Attributes:
        client (openai.AsyncOpenAI): The OpenAI async client instance
    """
    
    def __init__(self, client: openai.AsyncOpenAI):
        """
        Initialize the OpenAI provider with a client.
        
        Args:
            client (openai.AsyncOpenAI): The OpenAI async client instance
        """
        self.client = client

    async def stream(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        tools: list[Dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream responses from OpenAI's chat completion API.
        
        This method creates a streaming connection to OpenAI and yields
        StreamEvent objects containing text content and tool calls.
        
        Args:
            model (str): The OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            messages (list[ChatCompletionMessageParam]): Conversation messages
            tools (list[Dict[str, Any]] | None, optional): Available tools for the model. Defaults to None.
            **kwargs: Additional OpenAI API parameters
        
        Yields:
            StreamEvent: Events containing 'text' or 'tool_calls' data
        """
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
        """
        Generate a complete response from OpenAI's chat completion API.
        
        Args:
            model (str): The OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            messages (list[ChatCompletionMessageParam]): Conversation messages
            tools (list[Dict[str, Any]] | None, optional): Available tools for the model. Defaults to None.
            **kwargs: Additional OpenAI API parameters
        
        Returns:
            ChatCompletion: The complete response from OpenAI
        """
        completion = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            tools=tools,
            **kwargs,
        )
        return completion

    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Format tools for OpenAI's API format.
        
        Args:
            tools (List[Tool]): List of Tool instances to format
        
        Returns:
            List[Dict[str, Any]]: Tools formatted for OpenAI's API
        """
        return [tool.as_openai_tool() for tool in tools]

    async def _execute_single_tool(
        self, tool_call: Any, tool_map: Dict[str, Tool]
    ) -> Dict[str, Any] | None:
        """
        Execute a single tool call and return the result.
        
        This method handles both dict and ChatCompletionMessageToolCall object formats,
        executes the appropriate tool function, and returns the formatted result.
        
        Args:
            tool_call (Any): Tool call object or dict from OpenAI response
            tool_map (Dict[str, Tool]): Mapping of tool names to Tool instances
        
        Returns:
            Dict[str, Any] | None: Formatted tool result or None if tool not found
        """
        # Handle both dict and ChatCompletionMessageToolCall object formats
        if hasattr(tool_call, "function"):  # ChatCompletionMessageToolCall object
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id
        else:  # Dict format (fallback)
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

            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result),
            }
        else:
            logger.warning(f"Tool '{tool_name}' not found in tool_map")
            return None

    async def process_tool_calls(
        self,
        tool_calls: List[
            Any
        ],  # Changed from Dict to Any to handle ChatCompletionMessageToolCall objects
        tool_map: Dict[str, Tool],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple tool calls in parallel and return their results.
        
        This method executes all tool calls concurrently using asyncio.gather
        for improved performance, handling both successful executions and errors.
        
        Args:
            tool_calls (List[Any]): List of tool call objects from OpenAI response
            tool_map (Dict[str, Tool]): Mapping of tool names to Tool instances
        
        Returns:
            List[Dict[str, Any]]: List of formatted tool results
        """
        import asyncio
        
        if not tool_calls:
            return []

        # Execute all tool calls in parallel
        tasks = [
            self._execute_single_tool(tool_call, tool_map) for tool_call in tool_calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        tool_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Tool execution failed: {result}")
            elif result is not None:
                tool_results.append(result)

        return tool_results
