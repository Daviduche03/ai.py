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


class GoogleProvider(BaseProvider):
    """
    Google Generative AI provider implementation for the AI SDK.
    
    This class handles communication with Google's Gemini API, including streaming
    responses, generating completions, and processing tool calls.
    
    Attributes:
        client (genai.Client): The Google Generative AI client instance
    """
    
    def __init__(self, client: genai.Client):
        """
        Initialize the Google provider with a client.
        
        Args:
            client (genai.Client): The Google Generative AI client instance
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
        Stream responses from Google's Generative AI API.
        
        This method converts OpenAI-format messages to Google's format,
        creates a streaming connection, and yields StreamEvent objects.
        
        Args:
            model (str): The Google model to use (e.g., 'gemini-pro')
            messages (list[ChatCompletionMessageParam]): Conversation messages in OpenAI format
            tools (list[Dict[str, Any]] | None, optional): Available tools for the model. Defaults to None.
            **kwargs: Additional Google API parameters
        
        Yields:
            StreamEvent: Events containing 'text' or 'tool_calls' data
        
        Raises:
            ValueError: If system instruction is missing or no valid messages provided
        """
        contents = []
        system_instruction = ""

        for msg in messages:
            if msg is None:
                continue
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            
            if role == "system" and content:
                system_instruction = content
                continue
            elif role == "assistant":
                # Handle assistant messages with tool calls
                if tool_calls:
                    # For Google, we need to represent tool calls as function calls
                    # Skip adding this message as Google will generate the tool calls
                    continue
                elif content:
                    # Regular assistant message with content
                    contents.append({"role": "model", "parts": [{"text": content}]})
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
            elif role == "user" and content:
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role and content:
                # Default unknown roles to user
                contents.append({"role": "user", "parts": [{"text": content}]})

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
        """
        Generate a complete response from Google's Generative AI API.
        
        This method converts OpenAI-format messages to Google's format
        and generates a complete response.
        
        Args:
            model (str): The Google model to use (e.g., 'gemini-pro')
            messages (list[ChatCompletionMessageParam]): Conversation messages in OpenAI format
            tools (list[Dict[str, Any]] | None, optional): Available tools for the model. Defaults to None.
            **kwargs: Additional Google API parameters
        
        Returns:
            ChatCompletion: The complete response from Google
        
        Raises:
            ValueError: If system instruction is missing or no valid messages provided
        """
        contents = []
        system_instruction = ""

        for msg in messages:
            if msg is None:
                continue
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            
            if role == "system" and content:
                system_instruction = content
                continue
            elif role == "assistant":
                # Handle assistant messages with tool calls
                if tool_calls:
                    # For Google, we need to represent tool calls as function calls
                    # Skip adding this message as Google will generate the tool calls
                    continue
                elif content:
                    # Regular assistant message with content
                    contents.append({"role": "model", "parts": [{"text": content}]})
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
            elif role == "user" and content:
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role and content:
                # Default unknown roles to user
                contents.append({"role": "user", "parts": [{"text": content}]})

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
        """
        Format tools for Google's API format.
        
        Args:
            tools (List[Tool]): List of Tool instances to format
        
        Returns:
            List[Dict[str, Any]] | None: Tools formatted for Google's API, or None if no tools
        """
        if not tools:
            return None
        return [tool.as_google_tool() for tool in tools]

    async def _execute_single_tool(
        self, tool_call: Dict[str, Any], tool_map: Dict[str, Tool]
    ) -> Dict[str, Any] | None:
        """
        Execute a single tool call and return the result.
        
        This method extracts tool information from Google's format,
        executes the appropriate tool function, and returns the formatted result.
        
        Args:
            tool_call (Dict[str, Any]): Tool call object from Google response
            tool_map (Dict[str, Tool]): Mapping of tool names to Tool instances
        
        Returns:
            Dict[str, Any] | None: Formatted tool result or None if tool not found
        """
        tool_name = tool_call.name
        tool_args = tool_call.args

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
                "tool_call_id": random.randint(
                    1000, 9999
                ),  # Simulating a tool call ID
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result),
            }
        else:
            logger.warning(f"Tool '{tool_name}' not found in tool_map")
            return None

    async def process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_map: Dict[str, Tool],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple tool calls in parallel and return their results.
        
        This method executes all tool calls concurrently using asyncio.gather
        for improved performance, handling both successful executions and errors.
        
        Args:
            tool_calls (List[Dict[str, Any]]): List of tool call objects from Google response
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
