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
    """
    Google Generative AI provider implementation for the AI SDK.
    
    This class handles communication with Google's Gemini API, including streaming
    responses, generating completions, and processing tool calls.
    
    Attributes:
        client (genai.Client): The Google Generative AI client instance
        _message_cache (dict): Cache for converted messages to avoid reprocessing
    """
    
    def __init__(self, client: genai.Client):
        """
        Initialize the Google provider with a client.
        
        Args:
            client (genai.Client): The Google Generative AI client instance
        """
        self.client = client
        self._message_cache = {}

    def _convert_messages_to_google_format(
        self, messages: list[ChatCompletionMessageParam]
    ) -> tuple[list[Dict[str, Any]], str]:
        """
        Convert OpenAI-format messages to Google's format.
        
        This method is shared between stream() and generate() to avoid code duplication.
        
        Args:
            messages (list[ChatCompletionMessageParam]): Messages in OpenAI format
            
        Returns:
            tuple[list[Dict[str, Any]], str]: (contents, system_instruction)
            
        Raises:
            ValueError: If system instruction is missing or no valid messages provided
        """
        # Create cache key from messages
        cache_key = hash(str(messages))
        if cache_key in self._message_cache:
            return self._message_cache[cache_key]
            
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

        if not system_instruction:
            raise ValueError("System instruction must be provided.")

        # Ensure we have at least some content
        if not contents:
            raise ValueError("At least one user or assistant message is required.")
            
        result = (contents, system_instruction)
        
        # Cache result (limit cache size to prevent memory leaks)
        if len(self._message_cache) > 100:
            # Remove oldest entry
            oldest_key = next(iter(self._message_cache))
            del self._message_cache[oldest_key]
        self._message_cache[cache_key] = result
        
        return result

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
        try:
            contents, system_instruction = self._convert_messages_to_google_format(messages)
            
            config_kwargs = {"system_instruction": system_instruction}
            if tools:
                gemini_tools = types.Tool(function_declarations=tools)
                config_kwargs["tools"] = [gemini_tools]

            stream = self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            
            for chunk in stream:
                if chunk.text:
                    yield StreamEvent(event="text", data=chunk.text)
                if chunk.function_calls:
                    yield StreamEvent(event="tool_calls", data=chunk.function_calls)

        except Exception as e:
            logger.error(f"Error generating content with Google AI: {e}")
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
        try:
            contents, system_instruction = self._convert_messages_to_google_format(messages)
            
            config_kwargs = {"system_instruction": system_instruction}
            if tools:
                gemini_tools = types.Tool(function_declarations=tools)
                config_kwargs["tools"] = [gemini_tools]

            completion = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return completion
        except Exception as e:
            logger.error(f"Error generating content with Google AI: {e}")
            raise

    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [tool.as_google_tool() for tool in tools] if tools else None

    async def _execute_single_tool(
        self, tool_call: Dict[str, Any], tool_map: Dict[str, Tool]
    ) -> Dict[str, Any] | None:
        tool_name = tool_call.name
        tool_args = tool_call.args
        tool = tool_map.get(tool_name)

        if not tool:
            logger.warning(f"Tool '{tool_name}' not found")
            return None

        try:
            if inspect.iscoroutinefunction(tool.execute):
                result = await tool.execute(tool.parameters(**tool_args))
            else:
                result = tool.execute(tool.parameters(**tool_args))

            return {
                "tool_call_id": f"google_{random.randint(1000, 9999)}",
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result),
            }
        except Exception as e:
            logger.error(f"Tool execution failed for '{tool_name}': {e}")
            return None

    async def process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_map: Dict[str, Tool],
    ) -> List[Dict[str, Any]]:
        if not tool_calls:
            return []

        import asyncio
        
        tasks = [self._execute_single_tool(tool_call, tool_map) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if r is not None and not isinstance(r, Exception)]
