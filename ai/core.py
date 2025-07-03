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





async def streamText(
    model: LanguageModel,
    systemMessage: str,
    tools: list[Tool] = [],
    prompt: str = NOT_PROVIDED,
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

    if (
        "messages" in kwargs
        and len(kwargs["messages"]) > 0
        and prompt != NOT_PROVIDED
    ):
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
        async for event in provider.stream(
            model=model.model,
            **kwargs,
        ):
            # print("even: ", event)
            if event.event == 'text':
                yield f'0:{json.dumps(event.data)}\n'
            elif event.event == 'tool_calls':
                tool_calls = event.data
                tool_results = await provider.process_tool_calls(tool_calls, tool_map ) if tools else None


            # elif event.event == 'tool_calls':
            #     tool_calls = event.data
            #     tool_results = []
            #     for tool_call in tool_calls:
            #         tool_name = tool_call['function']['name']
            #         tool_args = json.loads(tool_call['function']['arguments'])
            #         tool = tool_map.get(tool_name)
            #         if tool:
            #             if inspect.iscoroutinefunction(tool.execute):
            #                 result = await tool.execute(tool.parameters(**tool_args))
            #             else:
            #                 result = tool.execute(tool.parameters(**tool_args))

            #             logger.info(f"Tool '{tool_name}' called with args {tool_args}. Result: {result}")
                        
            #             tool_results.append({
            #                 "tool_call_id": tool_call['id'],
            #                 "result": json.dumps(result)
            #             })
                
                kwargs['messages'].append({
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': tool_calls
                })
                for tool_result in tool_results:
                    kwargs['messages'].append({
                        'role': 'tool',
                        'tool_call_id': tool_result['tool_call_id'],
                        'content': tool_result['result']
                    })

                if 'tools' in kwargs:
                    del kwargs['tools']

                async for chunk in streamText(model, systemMessage, tools=tools, **kwargs):
                    yield chunk

    except Exception as e:
        logger.exception("Error during text generation")
        raise RuntimeError(f"Text generation failed: {str(e)}") from e


async def generateText(
    model: LanguageModel,
    systemMessage: str,
    tools: list[Tool] = [],
    prompt: str = NOT_PROVIDED,
    max_tool_calls: int = 5,
    **kwargs: Any,
) -> str:
    ProviderClass = PROVIDER_CLASSES.get(model.provider)
    if not ProviderClass:
        raise ValueError(f"Provider '{model.provider}' not found.")
    provider = ProviderClass(model.client)

    if "options" in kwargs:
        options = kwargs.pop("options")
        if isinstance(options, dict):
            kwargs.update(options)

    if (
        "messages" in kwargs
        and len(kwargs["messages"]) > 0
        and prompt != NOT_PROVIDED
    ):
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
        message = 'j'
        tool_calls = []
        if model.provider == 'google':
            message_text = completion.text
            message = {"role": "assistant", "content": message_text}
            tool_calls = completion.function_calls
        elif model.provider == 'openai':
            message = completion.choices[0].message
            tool_calls = message.tool_calls if hasattr(message, 'tool_calls') else []
        
        print("response: ", message, tool_calls)
        formatted_tools = await provider.process_tool_calls(tool_calls, tool_map ) if tools else None
        print("formatted_tools: ", formatted_tools)

        kwargs['messages'].append(message)
        if formatted_tools:
            for tool_result in formatted_tools:
                kwargs['messages'].append({
                        'role': 'tool',
                        'tool_call_id': tool_result['tool_call_id'],
                        'content': tool_result['result']
                    })





        # if tool_calls:
        #     tool_results = []
        #     for tool_call in tool_calls:
        #         tool_name = tool_call.function.name
        #         tool_args = json.loads(tool_call.function.arguments)
        #         tool = tool_map.get(tool_name)
        #         if tool:
        #             if inspect.iscoroutinefunction(tool.execute):
        #                 result = await tool.execute(tool.parameters(**tool_args))
        #             else:
        #                 result = tool.execute(tool.parameters(**tool_args))

        #             logger.info(f"Tool '{tool_name}' called with args {tool_args}. Result: {result}")
        #             tool_results.append({
        #                 "tool_call_id": tool_call.id,
        #                 "result": json.dumps(result)
        #             })
            
        #     kwargs['messages'].append(message)
        #     for tool_result in tool_results:
        #         kwargs['messages'].append({
        #             'role': 'tool',
        #             'tool_call_id': tool_result['tool_call_id'],
        #             'content': tool_result['result']
        #         })

        # Only recurse if we actually have tool calls to process and haven't exceeded max calls
        if formatted_tools and tool_calls and max_tool_calls > 0:
            if 'tools' in kwargs:
                del kwargs['tools']
            return await generateText(model, systemMessage, tools=tools, max_tool_calls=max_tool_calls-1, **kwargs)

        if model.provider == 'google':
            return message_text or ''
        elif model.provider == 'openai':
            return message.content if hasattr(message, 'content') else ''
        else:
            return message or ''

    except Exception as e:
        logger.exception("Error during text generation")
        raise RuntimeError(f"Text generation failed: {str(e)}") from e
