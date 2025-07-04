from typing import TypedDict, List, Optional, Literal, Union, Any, Awaitable, Callable

class TokenUsage(TypedDict):
    promptTokens: int
    completionTokens: int
    totalTokens: int

class ReasoningDetail(TypedDict):
    type: Literal['text', 'redacted']
    text: Optional[str]
    signature: Optional[str]
    data: Optional[str]

class Source(TypedDict):
    sourceType: Literal['url']
    id: str
    url: str
    title: Optional[str]
    providerMetadata: Optional[Any] # LanguageModelV1ProviderMetadata

class GeneratedFile(TypedDict):
    base64: str
    uint8Array: bytes # Assuming Uint8Array translates to bytes
    mimeType: str

class ToolCall(TypedDict):
    # Based on OpenAI's tool call structure
    id: str
    type: Literal['function']
    function: dict[str, str]

class ToolResult(TypedDict):
    tool_call_id: str
    result: Any

class Warning(TypedDict):
    # Assuming a simple structure
    message: str

class Response(TypedDict):
    id: str
    model: str
    timestamp: str # Using string for timestamp for simplicity
    headers: Optional[dict[str, str]]

class ResponseMessage(TypedDict):
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]]

class StepResult(TypedDict):
    # This is a bit vague, I'll assume a simple structure for now
    step: int
    tool_calls: Optional[List[ToolCall]]
    tool_results: Optional[List[ToolResult]]

class OnFinishResult(TypedDict):
    finishReason: Literal["stop", "length", "content-filter", "tool-calls", "error", "other", "unknown"]
    usage: TokenUsage
    providerMetadata: Optional[dict[str, dict[str, Any]]]
    text: str
    reasoning: Optional[str]
    reasoningDetails: List[ReasoningDetail]
    sources: List[Source]
    files: List[GeneratedFile]
    toolCalls: List[ToolCall]
    toolResults: List[ToolResult]
    warnings: Optional[List[Warning]]
    response: Response
    messages: List[ResponseMessage]
    steps: List[StepResult]

OnFinish = Callable[[OnFinishResult], Union[Awaitable[None], None]]
