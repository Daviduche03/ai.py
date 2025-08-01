"""
Python AI SDK - A streaming-first AI SDK inspired by the Vercel AI SDK.

This package provides a unified interface for working with multiple AI providers
with a focus on streaming responses and strict typing.
"""

from ai.core import generateText, streamText
from ai.model import LanguageModel, openai, google
from ai.tools import Tool
from ai.types import (
    TokenUsage,
    ReasoningDetail,
    OnFinish,
    OnFinishResult,
)

__version__ = "0.0.2"
__all__ = [
    # Core functions
    "generateText",
    "streamText", 
    # Classes
    "LanguageModel",
    "Tool",
    # Model helpers
    "openai",
    "google",
    # Types
    "TokenUsage",
    "ReasoningDetail", 
    "OnFinish",
    "OnFinishResult",
]