from typing import AsyncGenerator, Any
from ai.providers.openai import OpenAIProvider, StreamEvent
from ai.providers.google import GoogleProvider
from openai.types.chat import ChatCompletion
from ai.model import LanguageModel
from ai.tools import Tool
import json

import logging
import inspect

logger = logging.getLogger(__name__)

PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}

NOT_PROVIDED = "NOT_PROVIDED"


from ai.types import OnFinish, OnFinishResult


async def streamText(
    model: LanguageModel,
    systemMessage: str,
    tools: list[Tool] = [],
    prompt: str = NOT_PROVIDED,
    onFinish: OnFinish = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    ProviderClass = PROVIDER_CLASSES.get(model.provider)
    if not ProviderClass:
        raise ValueError(f"Provider '{model.provider}' not found.")
    provider = ProviderClass(model.client)

    if "options" in kwargs:
        options = kwargs.pop("options")
        if isinstance(options, dict):
            kwargs.update(options)

    if "messages" in kwargs and len(kwargs["messages"]) > 0 and prompt != NOT_PROVIDED:
        raise ValueError("Cannot use both 'messages' and 'prompt' together.")

    if prompt != NOT_PROVIDED:
        kwargs["messages"] = [{"role": "user", "content": prompt}]

    if not systemMessage:
        raise ValueError("System message must be provided.")

    kwargs["messages"].insert(0, {"role": "system", "content": systemMessage})

    if tools:
        kwargs["tools"] = provider.format_tools(tools)

    tool_map = {tool.name: tool for tool in tools}

    # Variables to track for onFinish
    full_response = ""
    all_tool_calls = []
    all_tool_results = []

    try:
        async for event in provider.stream(
            model=model.model,
            **kwargs,
        ):
            # print("even: ", event)
            if event.event == "text":
                full_response += event.data
                yield f"0:{json.dumps(event.data)}\n"
            elif event.event == "tool_calls":
                tool_calls = event.data
                all_tool_calls.extend(tool_calls)
                tool_results = (
                    await provider.process_tool_calls(tool_calls, tool_map)
                    if tools
                    else None
                )
                if tool_results:
                    all_tool_results.extend(tool_results)

                kwargs["messages"].append(
                    {"role": "assistant", "content": "", "tool_calls": tool_calls}
                )
                for tool_result in tool_results:
                    kwargs["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result["tool_call_id"],
                            "content": tool_result["content"],
                        }
                    )

                if "tools" in kwargs:
                    del kwargs["tools"]

                async for chunk in streamText(
                    model, systemMessage, tools=tools, onFinish=onFinish, **kwargs
                ):
                    yield chunk

        if onFinish:
            result = OnFinishResult(
                finishReason="stop",
                usage={
                    "promptTokens": 0,
                    "completionTokens": 0,
                    "totalTokens": 0,
                },  # Default usage
                providerMetadata=None,
                text=full_response,
                reasoning=None,
                reasoningDetails=[],
                sources=[],
                files=[],
                toolCalls=all_tool_calls,
                toolResults=all_tool_results,
                warnings=None,
                response={
                    "id": "",
                    "model": model.model,
                    "timestamp": "",
                    "headers": None,
                },
                messages=[],
                steps=[],
            )
            await onFinish(result)

    except Exception as e:
        logger.exception("Error during text generation")
        raise RuntimeError(f"Text generation failed: {str(e)}") from e


async def generateText(
    model: LanguageModel,
    systemMessage: str,
    tools: list[Tool] = [],
    prompt: str = NOT_PROVIDED,
    max_tool_calls: int = 5,
    onFinish: OnFinish = None,
    _accumulated_tool_calls: list = None,
    _accumulated_tool_results: list = None,
    **kwargs: Any,
) -> str:
    ProviderClass = PROVIDER_CLASSES.get(model.provider)
    if not ProviderClass:
        raise ValueError(f"Provider '{model.provider}' not found.")
    provider = ProviderClass(model.client)

    # Track all tool calls and results for onFinish callback
    all_tool_calls = _accumulated_tool_calls or []
    all_tool_results = _accumulated_tool_results or []

    if "options" in kwargs:
        options = kwargs.pop("options")
        if isinstance(options, dict):
            kwargs.update(options)

    if "messages" in kwargs and len(kwargs["messages"]) > 0 and prompt != NOT_PROVIDED:
        raise ValueError("Cannot use both 'messages' and 'prompt' together.")

    if prompt != NOT_PROVIDED:
        kwargs["messages"] = [{"role": "user", "content": prompt}]

    if not systemMessage:
        raise ValueError("System message must be provided.")

    kwargs["messages"].insert(0, {"role": "system", "content": systemMessage})

    if tools:
        kwargs["tools"] = provider.format_tools(tools)

    tool_map = {tool.name: tool for tool in tools}

    try:
        completion = await provider.generate(
            model=model.model,
            **kwargs,
        )
        # print(completion.text)
        message = "j"
        tool_calls = []
        if model.provider == "google":
            message_text = completion.text
            message = {"role": "assistant", "content": message_text}
            tool_calls = completion.function_calls
        elif model.provider == "openai":
            message = completion.choices[0].message
            tool_calls = message.tool_calls if hasattr(message, "tool_calls") else []

        # Track tool calls and results for onFinish callback BEFORE processing
        if tool_calls:
            all_tool_calls.extend(tool_calls)

        formatted_tools = (
            await provider.process_tool_calls(tool_calls, tool_map) if tools else None
        )

        if formatted_tools:
            all_tool_results.extend(formatted_tools)

        kwargs["messages"].append(message)

        if formatted_tools:
            for tool_result in formatted_tools:
                kwargs["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result["tool_call_id"],
                        "content": tool_result["content"],
                    }
                )

        # Only recurse if we actually have tool calls to process and haven't exceeded max calls
        if formatted_tools and tool_calls and max_tool_calls > 0:
            if "tools" in kwargs:
                del kwargs["tools"]
            return await generateText(
                model,
                systemMessage,
                tools=tools,
                max_tool_calls=max_tool_calls - 1,
                onFinish=onFinish,
                _accumulated_tool_calls=all_tool_calls,
                _accumulated_tool_results=all_tool_results,
                **kwargs,
            )

        if onFinish:
            # Extract finish reason based on provider
            finish_reason = "stop"
            usage_info = {"promptTokens": 0, "completionTokens": 0, "totalTokens": 0}
            provider_metadata = None

            if (
                model.provider == "openai"
                and hasattr(completion, "choices")
                and completion.choices
            ):
                finish_reason = completion.choices[0].finish_reason or "stop"
                if hasattr(completion, "usage") and completion.usage:
                    usage_info = {
                        "promptTokens": completion.usage.prompt_tokens,
                        "completionTokens": completion.usage.completion_tokens,
                        "totalTokens": completion.usage.total_tokens,
                    }

            # Get text content based on provider
            text_content = ""
            if model.provider == "google":
                text_content = message_text or ""
            elif model.provider == "openai":
                text_content = message.content if hasattr(message, "content") else ""

            result = OnFinishResult(
                finishReason=finish_reason,
                usage=usage_info,
                providerMetadata=provider_metadata,
                text=text_content,
                reasoning=None,
                reasoningDetails=[],
                sources=[],
                files=[],
                toolCalls=all_tool_calls,
                toolResults=all_tool_results,
                warnings=None,
                response={
                    "id": "",
                    "model": model.model,
                    "timestamp": "",
                    "headers": None,
                },
                messages=[],
                steps=[],
            )
            await onFinish(result)

        if model.provider == "google":
            return message_text or ""
        elif model.provider == "openai":
            return message.content if hasattr(message, "content") else ""
        else:
            return message or ""

    except Exception as e:
        logger.exception("Error during text generation")
        raise RuntimeError(f"Text generation failed: {str(e)}") from e
