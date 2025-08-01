from typing import Callable, Awaitable, Any, Dict, Type, Union, Optional
from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class Tool:
    """
    Represents a tool that can be called by AI models during generation.
    
    This class encapsu shouldn'tlates the metadata and functionality needed for AI models
    to understand and execute tools/functions during conversations.
    
    Attributes:
        name (str): The name of the tool/function
        description (str): Human-readable description of what the tool does
        parameters (Type[BaseModel]): Pydantic model defining the tool's parameters
        execute (Optional[Union[Callable[[BaseModel], Awaitable[Any]], Callable[[BaseModel], Any]]]): The actual function to execute when the tool is called. If None, the tool is client-side only.
    """
    name: str
    description: str
    parameters: Type[BaseModel]
    execute: Optional[Union[Callable[[BaseModel], Awaitable[Any]], Callable[[BaseModel], Any]]] = None  # Execution handler

    def as_openai_tool(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI's tool calling format.
        
        Returns:
            Dict[str, Any]: Tool definition in OpenAI's expected format with
                           'type' and 'function' fields
        """
        schema = self.parameters.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            }
        }

    def as_google_tool(self) -> Dict[str, Any]:
        """
        Convert the tool to Google's tool calling format.
        
        Returns:
            Dict[str, Any]: Tool definition in Google's expected format with
                           'name', 'description', and 'parameters' fields
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.model_json_schema(),
        }
