from typing import Dict, Any
import openai as OpenAI
from google import genai


class LanguageModel:
    def __init__(
        self, provider: str, model: str, client: any, options: Dict[str, Any] = {}
    ):
        self.provider = provider
        self.model = model
        self.client = client
        self.options = options


def openai(model: str, **kwargs: Any) -> LanguageModel:
    """
    Creates a LanguageModel instance for OpenAI.
    """
    client = OpenAI.AsyncOpenAI()
    return LanguageModel(provider="openai", model=model, client=client, options=kwargs)


def google(model: str, **kwargs: Any) -> LanguageModel:
    """
    Creates a LanguageModel instance for Google.
    """
    client = genai.Client(api_key="AIzaSyBumhLp15LJmcVhx4MssSBJOi8TAZc6k64")
    return LanguageModel(provider="google", model=model, client=client, options=kwargs)
