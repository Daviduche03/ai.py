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
        self._tool_call_buffer = {}

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

        self._tool_call_buffer.clear()

        async for chunk in stream:
           
            delta = chunk.choices[0].delta
            
            if delta.content:
                yield StreamEvent(event="text", data=delta.content)

            if delta.tool_calls:
                self._process_tool_call_chunks(delta.tool_calls)

        if self._tool_call_buffer:
            formatted_calls = self._format_buffered_tool_calls()
            if formatted_calls:
                yield StreamEvent(event="tool_calls", data=formatted_calls)

    def _process_tool_call_chunks(self, tool_call_chunks):
        for chunk in tool_call_chunks:
            idx = chunk.index
            if idx not in self._tool_call_buffer:
                self._tool_call_buffer[idx] = {"id": "", "name": "", "args": ""}
            
            if chunk.id:
                self._tool_call_buffer[idx]["id"] = chunk.id
            if chunk.function:
                if chunk.function.name:
                    self._tool_call_buffer[idx]["name"] = chunk.function.name
                if chunk.function.arguments:
                    self._tool_call_buffer[idx]["args"] += chunk.function.arguments

    def _format_buffered_tool_calls(self):
        return [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["args"],
                },
            }
            for call in self._tool_call_buffer.values()
            if call["id"] and call["name"]
        ]

    async def generate(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        tools: list[Dict[str, Any]] | None = None,
        **kwargs,
    ) -> ChatCompletion:
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            tools=tools,
            **kwargs,
        )

    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [tool.as_openai_tool() for tool in tools]

    async def _execute_single_tool(
        self, tool_call: Any, tool_map: Dict[str, Tool]
    ) -> Dict[str, Any] | None:
        try:
            if hasattr(tool_call, "function"):
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id
            else:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]

            tool = tool_map.get(tool_name)
            if not tool:
                logger.warning(f"Tool '{tool_name}' not found")
                return None

            if inspect.iscoroutinefunction(tool.execute):
                result = await tool.execute(tool.parameters(**tool_args))
            else:
                result = tool.execute(tool.parameters(**tool_args))

            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result),
            }
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return None

    async def process_tool_calls(
        self,
        tool_calls: List[Any],
        tool_map: Dict[str, Tool],
    ) -> List[Dict[str, Any]]:
        if not tool_calls:
            return []

        import asyncio
        
        tasks = [self._execute_single_tool(tool_call, tool_map) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    async def embed(self, model: str, input: str, **kwargs) -> List[float]:
        """Generate embeddings for a single text input using OpenAI."""
        response = await self.client.embeddings.create(
            model=model,
            input=input,
            **kwargs
        )
        return response.data[0].embedding

    async def embed_many(self, model: str, inputs: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple text inputs using OpenAI."""
        response = await self.client.embeddings.create(
            model=model,
            input=inputs,
            **kwargs
        )
        return [item.embedding for item in response.data]
