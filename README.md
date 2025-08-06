# Python AI SDK

A high-performance Python AI SDK inspired by Vercel AI SDK, built for production backends with streaming, multi-provider support, and type safety.

## Features

- **Streaming-first** - Real-time text generation with built-in streaming support
- **Multi-provider** - OpenAI, Google Gemini, and extensible architecture
- **Tool calling** - Server-side and client-side function execution
- **FastAPI ready** - Drop-in integration for web APIs
- **Type safe** - Full Pydantic validation and TypeScript-like experience
- **Analytics** - Built-in callbacks for monitoring and logging

## Installation

```bash
# Basic installation
pip install python-ai-sdk

# With FastAPI support
pip install python-ai-sdk[fastapi]

# With all optional dependencies
pip install python-ai-sdk[all]
```

## Quick Start

### Basic Text Generation

```python
from ai.core import generateText
from ai.model import openai
import os

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Generate text
response = await generateText(
    model=openai("gpt-4"),
    systemMessage="You are a helpful assistant.",
    prompt="What is the capital of France?"
)

print(response)  # "The capital of France is Paris."
```

### Streaming Text

```python
from ai.core import streamText
from ai.model import google

async for chunk in streamText(
    model=google("gemini-2.0-flash-exp"),
    systemMessage="You are a creative writer.",
    prompt="Write a short story about a robot."
):
    # chunk format: "0:{"text content"}\n"
    if chunk.startswith("0:"):
        import json
        text = json.loads(chunk[2:])
        print(text, end="", flush=True)
```

## Image Support

### Image from URL (Vercel AI SDK format)

```python
from ai import generateText, openai

# Simple and clean Vercel AI SDK format
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What do you see in this image?"},
        {
            "type": "image",
            "image": "https://example.com/image.jpg"  # URL string
        }
    ]
}

response = await generateText(
    model=openai("gpt-4o"),  # Vision-capable model
    systemMessage="You are an expert image analyst.",
    messages=[message]
)
```

### Image from File (Binary)

```python
import fs from 'fs'  # In Python: with open()

# Binary image data (like Vercel AI SDK)
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image"},
        {
            "type": "image", 
            "image": image_bytes  # Raw bytes
        }
    ]
}
```

### Base64 Images

```python
# Base64 string (no data URL prefix needed)
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image",
            "image": base64_string  # Just the base64 data
        }
    ]
}
```

### Helper Functions

```python
from ai.image import image_from_file, image_from_url

# Helper functions create the proper format
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Analyze these images:"},
        image_from_file("local_image.jpg"),      # Binary format
        image_from_url("https://example.com/img.png")  # URL format
    ]
}
```

### Multiple Images

```python
# Multiple images in one message
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Compare these images:"},
        {
            "type": "image",
            "image": "https://example.com/chart1.png"
        },
        {"type": "text", "text": "And this one:"},
        {
            "type": "image",
            "image": "https://example.com/chart2.png"
        },
        {"type": "text", "text": "What are the differences?"}
    ]
}
```

## Embeddings

### Single Text Embedding

```python
from ai import embed, openai_embedding

# Create embedding model
model = openai_embedding("text-embedding-3-small")

# Generate embedding
embedding = await embed(model, "Hello, world!")
print(f"Dimensions: {len(embedding)}")  # 1536 for text-embedding-3-small
print(f"First 5 values: {embedding[:5]}")
```

### Batch Embeddings

```python
from ai import embedMany, openai_embedding

model = openai_embedding("text-embedding-3-small")

texts = [
    "The cat sits on the mat",
    "Python is a programming language", 
    "Machine learning is fascinating"
]

embeddings = await embedMany(model, texts)
print(f"Generated {len(embeddings)} embeddings")
```

### Semantic Similarity

```python
import math

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))
    return dot_product / (magnitude_a * magnitude_b)

# Compare texts
model = openai_embedding("text-embedding-3-small")
embeddings = await embedMany(model, [
    "The cat sits on the mat",
    "A feline rests on the rug"  # Similar meaning
])

similarity = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.4f}")  # High similarity score
```

## Tool Calling

### Server-side Tools

```python
from ai.tools import Tool
from pydantic import BaseModel, Field

class WeatherParams(BaseModel):
    location: str = Field(..., description="City and country")

def get_weather(params: WeatherParams):
    # Your weather API logic here
    return {"location": params.location, "temperature": 22, "condition": "sunny"}

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters=WeatherParams,
    execute=get_weather
)

response = await generateText(
    model=openai("gpt-4"),
    systemMessage="You can check weather for users.",
    prompt="What's the weather in Tokyo?",
    tools=[weather_tool]
)
```

### Client-side Tools

```python
# Tool without execute function - handled by client
call_tool = Tool(
    name="make_call",
    description="Make a phone call",
    parameters=CallParams,
    # No execute - client handles this
)

# The AI will return tool calls for client to execute
async for chunk in streamText(
    model=openai("gpt-4"),
    systemMessage="You can make phone calls for users.",
    prompt="Call John at 555-0123",
    tools=[call_tool]
):
    if chunk.startswith("9:"):  # Tool call
        tool_call = json.loads(chunk[2:])
        print(f"Tool: {tool_call['toolName']}")
        print(f"Args: {tool_call['args']}")
```

## FastAPI Integration

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from ai.core import streamText
from ai.model import openai

app = FastAPI()

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    return StreamingResponse(
        streamText(
            model=openai("gpt-4"),
            systemMessage="You are a helpful assistant.",
            messages=messages,
            tools=[weather_tool]  # Optional tools
        ),
        media_type="text/plain; charset=utf-8"
    )

@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    
    response = await generateText(
        model=openai("gpt-4"),
        systemMessage="You are a helpful assistant.",
        prompt=body["prompt"]
    )
    
    return {"response": response}

@app.post("/api/embed")
async def create_embedding(request: Request):
    body = await request.json()
    
    embedding = await embed(
        model=openai_embedding("text-embedding-3-small"),
        value=body["text"]
    )
    
    return {"embedding": embedding, "dimensions": len(embedding)}

@app.post("/api/analyze-image")
async def analyze_image(request: Request):
    body = await request.json()
    
    message = create_image_message(
        body.get("prompt", "What do you see?"),
        body["image_url"]
    )
    
    response = await generateText(
        model=openai("gpt-4o"),
        systemMessage="You are an image analysis expert.",
        messages=[message]
    )
    
    return {"analysis": response}
```

## Analytics & Monitoring

```python
from ai.types import OnFinishResult

async def analytics_callback(result: OnFinishResult):
    print(f"Tokens used: {result['usage']['totalTokens']}")
    print(f"Finish reason: {result['finishReason']}")
    print(f"Tool calls: {len(result['toolCalls'])}")
    
    # Send to your analytics service
    # await send_to_analytics(result)

response = await generateText(
    model=openai("gpt-4"),
    systemMessage="You are helpful.",
    prompt="Hello!",
    onFinish=analytics_callback
)
```

## Supported Providers

### OpenAI

```python
from ai.model import openai, openai_embedding
import os

os.environ["OPENAI_API_KEY"] = "your-key"

# Chat models
model = openai("gpt-4o")           # GPT-4
model = openai("gpt-4.1")     # GPT-4.1

# Embedding models
embed_model = openai_embedding("text-embedding-3-small")  # 1536 dimensions
embed_model = openai_embedding("text-embedding-3-large")  # 3072 dimensions
```

### Google Gemini

```python
from ai.model import google, google_embedding
import os

os.environ["GOOGLE_API_KEY"] = "your-key"

# Chat models
model = google("gemini-2.5-pro")  # Latest Gemini


# Embedding models
embed_model = google_embedding("text-embedding-004")  # Latest embedding model
```

## Message Format

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]

response = await generateText(
    model=openai("gpt-4"),
    messages=messages  # Use messages instead of prompt
)
```

## Streaming Response Format

The streaming API returns formatted chunks:

```
f:{"messageId": "msg-abc123"}           # Message start
0:"Hello"                              # Text chunk
0:" there!"                            # More text
9:{"toolCallId":"call-1","toolName":"weather","args":{...}}  # Tool call
a:{"toolCallId":"call-1","result":"sunny"}  # Tool result
e:{"finishReason":"stop","usage":{...}} # Finish event
d:{"finishReason":"stop","usage":{...}} # Done
```

## Configuration

### Environment Variables

```bash
# Required for respective providers
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key

# Optional
OPENAI_BASE_URL=https://api.openai.com/v1  # Custom OpenAI endpoint
```

### Custom Client Configuration

```python
import openai
from ai.model import LanguageModel

# Custom OpenAI client
custom_client = openai.AsyncOpenAI(
    api_key="your-key",
    base_url="https://your-proxy.com/v1",
    timeout=30.0
)

model = LanguageModel(
    provider="openai",
    model="gpt-4",
    client=custom_client
)
```

## Development

```bash
# Clone the repository
git clone https://github.com/Daviduche03/ai.py
cd ai.py

# Install with Poetry
poetry install

# Install with pip (development mode)
pip install -e .

# Run tests
poetry run pytest

# Format code
poetry run ruff format

# Type checking
poetry run mypy ai/

# Run example
cd examples
python -m uvicorn fastapi_app:app --reload
```

## Examples

Check out the `/examples` directory for:
- FastAPI chat application with streaming
- Tool calling examples (server-side & client-side)
- Image analysis and vision capabilities
- Embedding generation and similarity search
- Multi-provider usage patterns
- Semantic search implementations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- [GitHub Repository](https://github.com/Daviduche03/ai.py)
- [PyPI Package](https://pypi.org/project/python-ai-sdk/)
- [Issues](https://github.com/Daviduche03/ai.py/issues)
