from typing import Callable, Awaitable, Any, Dict, Type, Union
from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    description: str
    parameters: Type[BaseModel]
    execute: Union[Callable[[BaseModel], Awaitable[Any]], Callable[[BaseModel], Any]]  # Execution handler

    def as_openai_tool(self) -> Dict[str, Any]:
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
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.model_json_schema(),
        }
