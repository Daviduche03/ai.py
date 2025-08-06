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
                if tool_calls:
                    # For Google, we need to represent tool calls as function calls
                    # Skip adding this message as Google will generate the tool calls
                    continue
                elif content:
                    contents.append({"role": "model", "parts": [{"text": content}]})
            elif role == "tool":
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
            elif role == "user":
                parts = self._process_user_content(content)
                if parts:
                    contents.append({"role": "user", "parts": parts})
            elif role and content:
                parts = self._process_user_content(content)
                if parts:
                    contents.append({"role": "user", "parts": parts})

        if not system_instruction:
            raise ValueError("System instruction must be provided.")

        if not contents:
            raise ValueError("At least one user or assistant message is required.")
            
        result = (contents, system_instruction)
        
        if len(self._message_cache) > 100:
            # Remove oldest entry
            oldest_key = next(iter(self._message_cache))
            del self._message_cache[oldest_key]
        self._message_cache[cache_key] = result
        
        return result

    def _process_user_content(self, content):
        """Process user content that can be text, images, files, or mixed."""
        if not content:
            return []
            
        if isinstance(content, str):
            return [{"text": content}]
        
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    
                    if item_type == "text":
                        parts.append({"text": item.get("text", "")})
                        
                    elif item_type == "image_url":
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "")
                        
                        # Handle base64 images
                        if url.startswith("data:image/"):
                            if ";base64," in url:
                                mime_type, base64_data = url.split(";base64,", 1)
                                mime_type = mime_type.replace("data:", "")
                                parts.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": base64_data
                                    }
                                })
                        else:
                            # Handle URL images (Google might not support direct URLs)
                            logger.warning("Google provider may not support image URLs directly")
                            parts.append({"text": f"[Image: {url}]"})
                            
                    elif item_type == "file":
                        # Handle file parts for Google
                        file_data = item.get("data", "")
                        media_type = item.get("mediaType", "application/octet-stream")
                        filename = item.get("filename", "file")
                        
                        if media_type.startswith("image/"):
                            parts.append({
                                "inline_data": {
                                    "mime_type": media_type,
                                    "data": file_data
                                }
                            })
                        else:
                            parts.append({"text": f"[File: {filename} ({media_type})]"})
                            
                elif isinstance(item, str):
                    parts.append({"text": item})
            return parts
        
        # Fallback for other types
        return [{"text": str(content)}]

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

    async def embed(self, model: str, input: str, **kwargs) -> List[float]:
        """Generate embeddings for a single text input using Google."""
        try:
            response = self.client.models.embed_content(
                model=model,
                contents=input,
                **kwargs
            )
            return response.embeddings
        except Exception as e:
            logger.error(f"Google embedding failed: {e}")
            raise

    async def embed_many(self, model: str, inputs: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple text inputs using Google."""
        try:
            import asyncio
            
            tasks = [self.embed(model, input_text, **kwargs) for input_text in inputs]
            results = await asyncio.gather(*tasks)
            return results
        except Exception as e:
            logger.error(f"Google batch embedding failed: {e}")
            raise
