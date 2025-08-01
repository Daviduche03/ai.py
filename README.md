# Python AI SDK

A Python AI SDK inspired by the Vercel AI SDK, designed for building AI-powered backends with a focus on streaming and strict typing.

## Features

- Streaming-first API
- Multi-provider support (OpenAI, Google, etc.)
- FastAPI integration
- Pydantic-based structured data generation

## Installation

### Basic Installation
```bash
pip install python-ai-sdk
```

### With Optional Dependencies
```bash
# For FastAPI integration
pip install python-ai-sdk[fastapi]

# For Google Generative AI support
pip install python-ai-sdk google-genai
```

### Installing from Test PyPI
If you want to test the latest development version:
```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ python-ai-sdk
```

## Quick Start

```python
from ai import generate_text, LanguageModel
import openai

# Configure your model
model = LanguageModel(
    provider="openai",
    model="gpt-4",
    client=openai.OpenAI(api_key="your-api-key")
)

# Generate text
result = await generate_text(
    model=model,
    prompt="What is the capital of France?"
)

print(result.text)
```

## Streaming Example

```python
from ai import stream_text

async for chunk in stream_text(
    model=model,
    prompt="Tell me a story"
):
    if chunk.type == "text-delta":
        print(chunk.textDelta, end="")
```

## Development

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run ruff format

# Type checking
poetry run mypy ai/
```
