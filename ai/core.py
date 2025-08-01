from typing import AsyncGenerator, Any
from ai.providers.openai import OpenAIProvider
from ai.providers.google import GoogleProvider
from ai.model import LanguageModel
from ai.tools import Tool
import json
import uuid
from ai.types import OnFinish, OnFinishResult
import logging


logger = logging.getLogger(__name__)

PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}

NOT_PROVIDED = "NOT_PROVIDED"


def _get_provider_class(provider_name: str):
    """Get provider class with helpful error messages."""
    ProviderClass = PROVIDER_CLASSES.get(provider_name)
    if not ProviderClass:
        available_providers = list(PROVIDER_CLASSES.keys())
        raise ValueError(f"Provider '{provider_name}' not supported. Available providers: {available_providers}")
    return ProviderClass





async def streamText(
    model: LanguageModel,
    systemMessage: str,
    tools: list[Tool] = [],
    prompt: str = NOT_PROVIDED,
    onFinish: OnFinish = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """
    Generate streaming text responses from AI models with optional tool calling support.
    
    This function provides real-time text generation with support for tool execution,
    allowing for interactive AI applications with dynamic capabilities.
    
    Args:
        model (LanguageModel): The language model instance to use for generation
        systemMessage (str): System message that defines the AI's behavior and context
        tools (list[Tool], optional): List of tools the AI can call during generation. Defaults to [].
        prompt (str, optional): User prompt. Cannot be used with 'messages' in kwargs. Defaults to NOT_PROVIDED.
        onFinish (OnFinish, optional): Callback function called when generation completes. Defaults to None.
        **kwargs: Additional arguments including:
            - messages: List of conversation messages (conflicts with prompt)
            - options: Dictionary of provider-specific options
            - Other provider-specific parameters
    
    Returns:
        AsyncGenerator[str, None]: Async generator yielding text chunks as they're generated
    
    Raises:
        ValueError: If provider is not found, both messages and prompt are provided,
                   or system message is missing
        RuntimeError: If text generation fails due to provider errors
    
    Example:
        ```python
        async for chunk in streamText(
            model=openai("gpt-4"),
            systemMessage="You are a helpful assistant.",
            prompt="Hello, how are you?"
        ):
            print(chunk, end="")
        ```
    """
    ProviderClass = _get_provider_class(model.provider)
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

    # Check for client-side tool results in messages
    client_tool_results = []
    for message in kwargs["messages"]:
        if message.get("role") == "assistant" and "toolInvocations" in message:
            for invocation in message["toolInvocations"]:
                if invocation.get("state") == "result":
                    client_tool_results.append({
                        "tool_call_id": invocation["toolCallId"],
                        "content": str(invocation["result"])
                    })
    
    # If we have client-side tool results, add them as tool messages
    if client_tool_results:
        # Find the assistant message with tool calls
        for i, message in enumerate(kwargs["messages"]):
            if (message.get("role") == "assistant" and 
                "toolInvocations" in message and 
                any(inv.get("state") == "result" for inv in message["toolInvocations"])):
                
                # Extract tool calls from toolInvocations
                tool_calls = []
                for invocation in message["toolInvocations"]:
                    if invocation.get("state") == "result":
                        tool_calls.append({
                            "id": invocation["toolCallId"],
                            "type": "function",
                            "function": {
                                "name": invocation["toolName"],
                                "arguments": json.dumps(invocation["args"])
                            }
                        })
                
                # Update the assistant message to include tool_calls
                kwargs["messages"][i] = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls
                }
                
                # Add tool result messages after the assistant message
                for j, result in enumerate(client_tool_results):
                    kwargs["messages"].insert(i + 1 + j, {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"]
                    })
                break

    if tools:
        kwargs["tools"] = provider.format_tools(tools)

    tool_map = {tool.name: tool for tool in tools}

    # Variables to track for onFinish
    full_response = ""
    all_tool_calls = []
    all_tool_results = []
    message_id = f"msg-{uuid.uuid4().hex[:24]}"
    
    # Yield initial message ID
    yield f"f:{json.dumps({"messageId": message_id})}\n"

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
                
                # Yield tool calls immediately
                for tool_call in tool_calls:
                    # Handle different provider formats
                    if hasattr(tool_call, 'name'):  # Google FunctionCall object
                        tool_call_data = {
                            "toolCallId": f"call_{uuid.uuid4().hex[:24]}",
                            "toolName": tool_call.name,
                            "args": tool_call.args
                        }
                        tool_name = tool_call.name
                    else:  # OpenAI format (dictionary)
                        tool_call_data = {
                            "toolCallId": tool_call["id"],
                            "toolName": tool_call["function"]["name"],
                            "args": json.loads(tool_call["function"]["arguments"])
                        }
                        tool_name = tool_call["function"]["name"]
                    
                    yield f"9:{json.dumps(tool_call_data)}\n"
                
                # Check if any tools have execute functions (server-side tools)
                has_server_side_tools = any(
                    hasattr(tool_map.get(
                        tool_call.name if hasattr(tool_call, 'name') else tool_call["function"]["name"]
                    ), 'execute') and 
                    tool_map.get(
                        tool_call.name if hasattr(tool_call, 'name') else tool_call["function"]["name"]
                    ).execute is not None
                    for tool_call in tool_calls
                    if tool_map.get(
                        tool_call.name if hasattr(tool_call, 'name') else tool_call["function"]["name"]
                    )
                )
                
                if has_server_side_tools:
                    # Process server-side tools
                    tool_results = (
                        await provider.process_tool_calls(tool_calls, tool_map)
                        if tools
                        else None
                    )
                    if tool_results:
                        all_tool_results.extend(tool_results)
                        
                        # Yield tool results immediately
                        for tool_result in tool_results:
                            result_data = {
                                "toolCallId": tool_result["tool_call_id"],
                                "result": tool_result["content"]
                            }
                            yield f"a:{json.dumps(result_data)}\n"

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

                    # Yield finish reason for tool calls step
                    yield f"e:{json.dumps({"finishReason": "tool-calls", "usage": {"promptTokens": 0, "completionTokens": 0}, "isContinued": True})}\n"

                    async for chunk in streamText(
                        model, systemMessage, tools=tools, onFinish=onFinish, **kwargs
                    ):
                        yield chunk
                    return
                else:
                    # Client-side tools only - finish here
                    yield f"e:{json.dumps({"finishReason": "tool-calls", "usage": {"promptTokens": 0, "completionTokens": 0}, "isContinued": False})}\n"
                    yield f"d:{json.dumps({"finishReason": "tool-calls", "usage": {"promptTokens": 0, "completionTokens": 0}})}\n"
                    return
        
        # Yield final finish event for non-tool-call completions
        yield f"e:{json.dumps({"finishReason": "stop", "usage": {"promptTokens": 0, "completionTokens": 0}, "isContinued": False})}\n"
        yield f"d:{json.dumps({"finishReason": "stop", "usage": {"promptTokens": 0, "completionTokens": len(full_response.split())}})}\n"

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
    """
    Generate complete text responses from AI models with tool calling and recursion support.
    
    This function generates a complete text response, handling tool calls recursively
    until the AI provides a final answer or reaches the maximum tool call limit.
    
    Args:
        model (LanguageModel): The language model instance to use for generation
        systemMessage (str): System message that defines the AI's behavior and context
        tools (list[Tool], optional): List of tools the AI can call during generation. Defaults to [].
        prompt (str, optional): User prompt. Cannot be used with 'messages' in kwargs. Defaults to NOT_PROVIDED.
        max_tool_calls (int, optional): Maximum number of recursive tool calls allowed. Defaults to 5.
        onFinish (OnFinish, optional): Callback function called when generation completes. Defaults to None.
        _accumulated_tool_calls (list, optional): Internal parameter for tracking tool calls across recursions. Defaults to None.
        _accumulated_tool_results (list, optional): Internal parameter for tracking tool results across recursions. Defaults to None.
        **kwargs: Additional arguments including:
            - messages: List of conversation messages (conflicts with prompt)
            - options: Dictionary of provider-specific options
            - Other provider-specific parameters
    
    Returns:
        str: The complete generated text response
    
    Raises:
        ValueError: If provider is not found, both messages and prompt are provided,
                   or system message is missing
        RuntimeError: If text generation fails due to provider errors or max recursion reached
    
    Example:
        ```python
        response = await generateText(
            model=google("gemini-pro"),
            systemMessage="You are a helpful assistant with access to tools.",
            prompt="What's the weather like in Paris?",
            tools=[weather_tool],
            max_tool_calls=3
        )
        print(response)
        ```
    """
    ProviderClass = _get_provider_class(model.provider)
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
