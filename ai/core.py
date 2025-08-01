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
    ProviderClass = PROVIDER_CLASSES.get(provider_name)
    if not ProviderClass:
        available = list(PROVIDER_CLASSES.keys())
        raise ValueError(f"Provider '{provider_name}' not supported. Available: {available}")
    return ProviderClass


def _has_server_side_execution(tool_call, tool_map):
    tool_name = tool_call.name if hasattr(tool_call, 'name') else tool_call["function"]["name"]
    tool = tool_map.get(tool_name)
    return tool and hasattr(tool, 'execute') and tool.execute is not None


def _process_client_tool_results(messages):
    client_tool_results = []
    for message in messages:
        if message.get("role") == "assistant" and "toolInvocations" in message:
            for invocation in message["toolInvocations"]:
                if invocation.get("state") == "result":
                    client_tool_results.append({
                        "tool_call_id": invocation["toolCallId"],
                        "content": str(invocation["result"])
                    })
    
    if client_tool_results:
        for i, message in enumerate(messages):
            if (message.get("role") == "assistant" and 
                "toolInvocations" in message and 
                any(inv.get("state") == "result" for inv in message["toolInvocations"])):
                
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
                
                messages[i] = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls
                }
                
                for j, result in enumerate(client_tool_results):
                    messages.insert(i + 1 + j, {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"]
                    })
                break





async def streamText(
    model: LanguageModel,
    systemMessage: str,
    tools: list[Tool] = [],
    prompt: str = NOT_PROVIDED,
    onFinish: OnFinish = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    provider = _get_provider_class(model.provider)(model.client)

    if "options" in kwargs:
        kwargs.update(kwargs.pop("options"))

    if "messages" in kwargs and kwargs["messages"] and prompt != NOT_PROVIDED:
        raise ValueError("Cannot use both 'messages' and 'prompt'")

    if prompt != NOT_PROVIDED:
        kwargs["messages"] = [{"role": "user", "content": prompt}]

    if not systemMessage:
        raise ValueError("System message required")

    kwargs["messages"].insert(0, {"role": "system", "content": systemMessage})

    _process_client_tool_results(kwargs["messages"])

    if tools:
        kwargs["tools"] = provider.format_tools(tools)

    tool_map = {tool.name: tool for tool in tools}
    full_response = ""
    all_tool_calls = []
    all_tool_results = []
    message_id = f"msg-{uuid.uuid4().hex[:24]}"
    
    yield f"f:{json.dumps({'messageId': message_id})}\n"

    try:
        async for event in provider.stream(model=model.model, **kwargs):
            if event.event == "text":
                full_response += event.data
                yield f"0:{json.dumps(event.data)}\n"
            elif event.event == "tool_calls":
                tool_calls = event.data
                all_tool_calls.extend(tool_calls)
                
                for tool_call in tool_calls:
                    if hasattr(tool_call, 'name'):
                        tool_call_data = {
                            "toolCallId": f"call_{uuid.uuid4().hex[:24]}",
                            "toolName": tool_call.name,
                            "args": tool_call.args
                        }
                        tool_name = tool_call.name
                    else:
                        tool_call_data = {
                            "toolCallId": tool_call["id"],
                            "toolName": tool_call["function"]["name"],
                            "args": json.loads(tool_call["function"]["arguments"])
                        }
                        tool_name = tool_call["function"]["name"]
                    
                    yield f"9:{json.dumps(tool_call_data)}\n"
                
                has_server_side_tools = any(
                    _has_server_side_execution(tool_call, tool_map)
                    for tool_call in tool_calls
                )
                
                if has_server_side_tools:
                    tool_results = await provider.process_tool_calls(tool_calls, tool_map) if tools else []
                    if tool_results:
                        all_tool_results.extend(tool_results)
                        
                        for tool_result in tool_results:
                            result_data = {
                                "toolCallId": tool_result["tool_call_id"],
                                "result": tool_result["content"]
                            }
                            yield f"a:{json.dumps(result_data)}\n"

                    kwargs["messages"].append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                    kwargs["messages"].extend([
                        {
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "content": result["content"],
                        }
                        for result in tool_results
                    ])

                    kwargs.pop("tools", None)

                    yield f"e:{json.dumps({'finishReason': 'tool-calls', 'usage': {'promptTokens': 0, 'completionTokens': 0}, 'isContinued': True})}\n"

                    async for chunk in streamText(model, systemMessage, tools=tools, onFinish=onFinish, **kwargs):
                        yield chunk
                    return
                else:
                    yield f"e:{json.dumps({'finishReason': 'tool-calls', 'usage': {'promptTokens': 0, 'completionTokens': 0}, 'isContinued': False})}\n"
                    yield f"d:{json.dumps({'finishReason': 'tool-calls', 'usage': {'promptTokens': 0, 'completionTokens': 0}})}\n"
                    return
        
        yield f"e:{json.dumps({'finishReason': 'stop', 'usage': {'promptTokens': 0, 'completionTokens': 0}, 'isContinued': False})}\n"
        yield f"d:{json.dumps({'finishReason': 'stop', 'usage': {'promptTokens': 0, 'completionTokens': len(full_response.split())}})}\n"

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
    provider = _get_provider_class(model.provider)(model.client)
    all_tool_calls = _accumulated_tool_calls or []
    all_tool_results = _accumulated_tool_results or []

    if "options" in kwargs:
        kwargs.update(kwargs.pop("options"))

    if "messages" in kwargs and kwargs["messages"] and prompt != NOT_PROVIDED:
        raise ValueError("Cannot use both 'messages' and 'prompt'")

    if prompt != NOT_PROVIDED:
        kwargs["messages"] = [{"role": "user", "content": prompt}]

    if not systemMessage:
        raise ValueError("System message required")

    kwargs["messages"].insert(0, {"role": "system", "content": systemMessage})

    if tools:
        kwargs["tools"] = provider.format_tools(tools)

    tool_map = {tool.name: tool for tool in tools}

    try:
        completion = await provider.generate(model=model.model, **kwargs)
        
        if model.provider == "google":
            message_text = completion.text
            message = {"role": "assistant", "content": message_text}
            tool_calls = completion.function_calls
        else:  # openai
            message = completion.choices[0].message
            tool_calls = message.tool_calls if hasattr(message, "tool_calls") else []

        if tool_calls:
            all_tool_calls.extend(tool_calls)

        formatted_tools = await provider.process_tool_calls(tool_calls, tool_map) if tools and tool_calls else []

        if formatted_tools:
            all_tool_results.extend(formatted_tools)

        kwargs["messages"].append(message)

        if formatted_tools:
            kwargs["messages"].extend([
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                }
                for result in formatted_tools
            ])

        if formatted_tools and tool_calls and max_tool_calls > 0:
            kwargs.pop("tools", None)
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
            finish_reason = "stop"
            usage_info = {"promptTokens": 0, "completionTokens": 0, "totalTokens": 0}

            if model.provider == "openai" and hasattr(completion, "choices") and completion.choices:
                finish_reason = completion.choices[0].finish_reason or "stop"
                if hasattr(completion, "usage") and completion.usage:
                    usage_info = {
                        "promptTokens": completion.usage.prompt_tokens,
                        "completionTokens": completion.usage.completion_tokens,
                        "totalTokens": completion.usage.total_tokens,
                    }

            text_content = (message_text if model.provider == "google" 
                          else getattr(message, "content", ""))

            result = OnFinishResult(
                finishReason=finish_reason,
                usage=usage_info,
                providerMetadata=None,
                text=text_content or "",
                reasoning=None,
                reasoningDetails=[],
                sources=[],
                files=[],
                toolCalls=all_tool_calls,
                toolResults=all_tool_results,
                warnings=None,
                response={"id": "", "model": model.model, "timestamp": "", "headers": None},
                messages=[],
                steps=[],
            )
            await onFinish(result)

        return (message_text if model.provider == "google" 
                else getattr(message, "content", "") or "")

    except Exception as e:
        logger.exception("Error during text generation")
        raise RuntimeError(f"Text generation failed: {str(e)}") from e
