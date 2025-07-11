from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List
from openai.types.chat import ChatCompletion
from ai.tools import Tool

class BaseProvider(ABC):
    """
    Abstract base class for AI provider implementations.
    
    This class defines the interface that all AI providers (OpenAI, Google, etc.)
    must implement to work with the AI SDK.
    """
    @abstractmethod
    async def stream(self, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream responses from the AI provider.
        
        Args:
            **kwargs: Provider-specific arguments for streaming
            
        Yields:
            Dict[str, Any]: Streaming response chunks
        """
        pass

    @abstractmethod
    async def generate(self, **kwargs) -> ChatCompletion:
        """
        Generate a complete response from the AI provider.
        
        Args:
            **kwargs: Provider-specific arguments for generation
            
        Returns:
            ChatCompletion: The complete response from the AI provider
        """
        pass

    @abstractmethod
    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Format tools for the specific AI provider.
        
        Args:
            tools (List[Tool]): List of tools to format
            
        Returns:
            List[Dict[str, Any]]: Provider-specific formatted tools
        """
        pass

    @abstractmethod
    def process_tool_calls(
        self, tool_calls: List[Dict[str, Any]], tool_map: Dict[str, Tool]
    ) -> List[Dict[str, Any]]:
        """
        Process tool calls from the AI provider response.
        
        Args:
            tool_calls (List[Dict[str, Any]]): Raw tool calls from the provider
            tool_map (Dict[str, Tool]): Mapping of tool names to Tool instances
            
        Returns:
            List[Dict[str, Any]]: Processed tool call results
        """
        pass