from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List
from openai.types.chat import ChatCompletion
from ai.tools import Tool

class BaseProvider(ABC):
    @abstractmethod
    async def stream(self, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        pass

    @abstractmethod
    async def generate(self, **kwargs) -> ChatCompletion:
        pass

    @abstractmethod
    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def process_tool_calls(
        self, tool_calls: List[Dict[str, Any]], tool_map: Dict[str, Tool]
    ) -> List[Dict[str, Any]]:
        pass