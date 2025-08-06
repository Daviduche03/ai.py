from typing import AsyncGenerator, Any, List, Union, Dict
import base64
import mimetypes
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


def _validate_and_convert_content(content):
    """Validate and convert content to provider-specific format."""
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        converted_content = []
        
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                
                if item_type == "text":
                    converted_content.append(item)
                    
                elif item_type == "image":
                    # Convert Vercel AI SDK format to OpenAI format
                    image_data = item.get("image")
                    
                    if isinstance(image_data, str):
                        # Handle URL or base64 string
                        if image_data.startswith(("http://", "https://")):
                            # URL format
                            converted_content.append({
                                "type": "image_url",
                                "image_url": {"url": image_data}
                            })
                        elif image_data.startswith("data:"):
                            # Data URL format
                            converted_content.append({
                                "type": "image_url", 
                                "image_url": {"url": image_data}
                            })
                        else:
                            # Plain base64 string - assume JPEG
                            try:
                                base64.b64decode(image_data, validate=True)
                                converted_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                                })
                            except Exception as e:
                                raise ValueError(f"Invalid base64 image data: {e}")
                                
                    elif isinstance(image_data, bytes):
                        # Binary image data - convert to base64
                        base64_data = base64.b64encode(image_data).decode("utf-8")
                        converted_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}
                        })
                    else:
                        raise ValueError(f"Invalid image data type: {type(image_data)}")
                        
                elif item_type == "file":
                    # Handle file parts (for providers that support them)
                    file_data = item.get("data")
                    media_type = item.get("mediaType", "application/octet-stream")
                    filename = item.get("filename")
                    
                    if isinstance(file_data, bytes):
                        # Convert to base64 for transmission
                        base64_data = base64.b64encode(file_data).decode("utf-8")
                        file_part = {
                            "type": "file",
                            "data": base64_data,
                            "mediaType": media_type
                        }
                        if filename:
                            file_part["filename"] = filename
                        converted_content.append(file_part)
                    else:
                        raise ValueError(f"Invalid file data type: {type(file_data)}")
                        
                elif item_type == "image_url":
                    # Already in OpenAI format
                    converted_content.append(item)
                    
                else:
                    # Unknown type, pass through
                    converted_content.append(item)
            else:
                # Non-dict item, pass through
                converted_content.append(item)
                
        return converted_content
    
    return content


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
    _accumulated_tool_calls: list = None,
    _accumulated_tool_results: list = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """
    Generate streaming text responses from AI models with tool calling support.
    
    This function provides real-time text generation with support for tool execution,
    allowing for interactive AI applications with dynamic capabilities. The response
    is streamed as formatted chunks that can be processed in real-time.
    
    Args:
        model (LanguageModel): The language model instance to use for generation
        systemMessage (str): System message that defines the AI's behavior and context
        tools (list[Tool], optional): List of tools the AI can call during generation. Defaults to [].
        prompt (str, optional): User prompt. Cannot be used with 'messages' in kwargs. Defaults to NOT_PROVIDED.
        onFinish (OnFinish, optional): Callback function called when generation completes. Defaults to None.
        _accumulated_tool_calls (list, optional): Internal parameter for tracking tool calls across recursions. Defaults to None.
        _accumulated_tool_results (list, optional): Internal parameter for tracking tool results across recursions. Defaults to None.
        **kwargs: Additional arguments including:
            - messages: List of conversation messages (conflicts with prompt)
            - options: Dictionary of provider-specific options
            - Other provider-specific parameters
    
    Returns:
        AsyncGenerator[str, None]: Async generator yielding formatted chunks:
            - f:{messageId} - Message start with unique ID
            - 0:"text" - Text content chunks
            - 9:{toolCall} - Tool call data
            - a:{toolResult} - Tool execution result
            - e:{finish} - Finish event with reason and usage
            - d:{done} - Final completion event
    
    Raises:
        ValueError: If provider is not found, both messages and prompt are provided,
                   or system message is missing
        RuntimeError: If text generation fails due to provider errors
    
    Example:
        ```python
        async for chunk in streamText(
            model=openai("gpt-4"),
            systemMessage="You are a helpful assistant.",
            prompt="Hello, how are you?",
            tools=[weather_tool]
        ):
            if chunk.startswith("0:"):
                text = json.loads(chunk[2:])
                print(text, end="", flush=True)
        ```
    """
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

    # Validate and convert content in messages
    for message in kwargs["messages"]:
        if message.get("role") in ["user", "assistant"]:
            message["content"] = _validate_and_convert_content(message.get("content"))

    _process_client_tool_results(kwargs["messages"])

    if tools:
        kwargs["tools"] = provider.format_tools(tools)

    tool_map = {tool.name: tool for tool in tools}
    full_response = ""
    all_tool_calls = _accumulated_tool_calls or []
    all_tool_results = _accumulated_tool_results or []
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

                    async for chunk in streamText(
                        model, 
                        systemMessage, 
                        tools=tools, 
                        onFinish=onFinish,
                        _accumulated_tool_calls=all_tool_calls,
                        _accumulated_tool_results=all_tool_results,
                        **kwargs
                    ):
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
                },
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
    Unlike streamText, this returns the complete response as a single string.
    
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

    # Validate and convert content in messages
    for message in kwargs["messages"]:
        if message.get("role") in ["user", "assistant"]:
            message["content"] = _validate_and_convert_content(message.get("content"))

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


async def embed(
    model: LanguageModel,
    value: str,
    **kwargs: Any,
) -> List[float]:
    """
    Generate embeddings for a single text input.
    
    This function creates vector embeddings from text using the specified model.
    Embeddings are useful for semantic search, similarity comparison, and other
    NLP tasks that require numerical representation of text.
    
    Args:
        model (LanguageModel): The language model to use for embeddings (must support embeddings)
        value (str): Text string to embed
        **kwargs: Additional provider-specific parameters
        
    Returns:
        List[float]: The embedding vector as a list of floating-point numbers
        
    Raises:
        ValueError: If provider doesn't support embeddings
        RuntimeError: If embedding generation fails
        
    Example:
        ```python
        model = openai_embedding("text-embedding-3-small")
        embedding = await embed(model, "Hello, world!")
        print(f"Embedding dimensions: {len(embedding)}")
        ```
    """
    provider = _get_provider_class(model.provider)(model.client)
    
    if not hasattr(provider, 'embed'):
        raise ValueError(f"Provider '{model.provider}' does not support embeddings")
    
    try:
        return await provider.embed(model=model.model, input=value, **kwargs)
    except Exception as e:
        logger.exception("Error during embedding generation")
        raise RuntimeError(f"Embedding generation failed: {str(e)}") from e


async def embedMany(
    model: LanguageModel,
    values: List[str],
    **kwargs: Any,
) -> List[List[float]]:
    """
    Generate embeddings for multiple text inputs efficiently.
    
    This function creates vector embeddings for multiple texts in a single request,
    which is more efficient than calling embed() multiple times. Useful for batch
    processing, building vector databases, or comparing multiple texts.
    
    Args:
        model (LanguageModel): The language model to use for embeddings (must support embeddings)
        values (List[str]): List of text strings to embed
        **kwargs: Additional provider-specific parameters
        
    Returns:
        List[List[float]]: List of embedding vectors, one for each input text
        
    Raises:
        ValueError: If provider doesn't support embeddings or values is empty
        RuntimeError: If embedding generation fails
        
    Example:
        ```python
        model = openai_embedding("text-embedding-3-small")
        texts = ["Hello world", "Python is great", "AI is amazing"]
        embeddings = await embedMany(model, texts)
        
        # Calculate similarity between first two texts
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        ```
    """
    if not values:
        raise ValueError("Values list cannot be empty")
        
    provider = _get_provider_class(model.provider)(model.client)
    
    if not hasattr(provider, 'embed_many'):
        raise ValueError(f"Provider '{model.provider}' does not support embeddings")
    
    try:
        return await provider.embed_many(model=model.model, inputs=values, **kwargs)
    except Exception as e:
        logger.exception("Error during batch embedding generation")
        raise RuntimeError(f"Batch embedding generation failed: {str(e)}") from e