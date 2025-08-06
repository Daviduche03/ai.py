"""
Python AI SDK - A streaming-first AI SDK inspired by the Vercel AI SDK.

This package provides a unified interface for working with multiple AI providers
with a focus on streaming responses and strict typing.
"""

from ai.core import generateText, streamText, embed, embedMany
from ai.model import LanguageModel, openai, google, openai_embedding, google_embedding
from ai.tools import Tool
from ai.types import (
    TokenUsage,
    ReasoningDetail,
    OnFinish,
    OnFinishResult,
)
from ai.image import (
    image_from_file,
    image_from_url,
    image_from_base64,
    image_from_bytes,
    download_and_encode_image,
    create_image_message,
    text_with_image,
    text_with_url_image,
    file_from_path,
    file_from_bytes,
    download_file,
)

__version__ = "0.0.2"
__all__ = [
    # Core functions
    "generateText",
    "streamText",
    "embed", 
    "embedMany",
    # Classes
    "LanguageModel",
    "Tool",
    # Model helpers
    "openai",
    "google",
    "openai_embedding",
    "google_embedding",
    # Image utilities
    "image_from_file",
    "image_from_url", 
    "image_from_base64",
    "image_from_bytes",
    "download_and_encode_image",
    "create_image_message",
    "text_with_image",
    "text_with_url_image",
    # File utilities
    "file_from_path",
    "file_from_bytes", 
    "download_file",
    # Types
    "TokenUsage",
    "ReasoningDetail", 
    "OnFinish",
    "OnFinishResult",
]