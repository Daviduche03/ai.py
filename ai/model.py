from typing import Dict, Any
import openai as OpenAI
from google import genai


class LanguageModel:
    """
    Represents a language model instance with provider-specific configuration.
    
    This class encapsulates the configuration needed to interact with different
    AI providers (OpenAI, Google, etc.) in a unified way.
    
    Attributes:
        provider (str): The name of the AI provider (e.g., 'openai', 'google')
        model (str): The specific model name (e.g., 'gpt-4', 'gemini-pro')
        client (any): The provider-specific client instance
        options (Dict[str, Any]): Additional provider-specific options
    """
    
    def __init__(
        self, provider: str, model: str, client: any, options: Dict[str, Any] = {}
    ):
        """
        Initialize a LanguageModel instance.
        
        Args:
            provider (str): The AI provider name
            model (str): The model identifier
            client (any): Provider-specific client instance
            options (Dict[str, Any], optional): Additional configuration options. Defaults to {}.
        """
        self.provider = provider
        self.model = model
        self.client = client
        self.options = options


def openai(model: str, **kwargs: Any) -> LanguageModel:
    """
    Create a LanguageModel instance configured for OpenAI models.
    
    This factory function initializes an OpenAI client and creates a LanguageModel
    instance ready for use with OpenAI's API.
    
    Args:
        model (str): The OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
        **kwargs: Additional options to pass to the model configuration
    
    Returns:
        LanguageModel: Configured LanguageModel instance for OpenAI
    
    Example:
        ```python
        model = openai("gpt-4", temperature=0.7, max_tokens=1000)
        ```
    
    Note:
        Requires OPENAI_API_KEY environment variable to be set.
    """
    client = OpenAI.AsyncOpenAI()
    return LanguageModel(provider="openai", model=model, client=client, options=kwargs)


def google(model: str, **kwargs: Any) -> LanguageModel:
    """
    Create a LanguageModel instance configured for Google Generative AI models.
    
    This factory function initializes a Google Generative AI client and creates a
    LanguageModel instance ready for use with Google's Gemini API.
    
    Args:
        model (str): The Google model name (e.g., 'gemini-pro', 'gemini-pro-vision')
        **kwargs: Additional options to pass to the model configuration
    
    Returns:
        LanguageModel: Configured LanguageModel instance for Google
    
    Example:
        ```python
        model = google("gemini-pro", temperature=0.7, max_output_tokens=1000)
        ```
    
    Note:
        Requires GOOGLE_API_KEY environment variable to be set.
    """
    client = genai.Client()
    return LanguageModel(provider="google", model=model, client=client, options=kwargs)

def openai_embedding(model: str = "text-embedding-3-small", **kwargs: Any) -> LanguageModel:
    """
    Create a LanguageModel instance configured for OpenAI embedding models.
    
    Args:
        model: The OpenAI embedding model name
        **kwargs: Additional options
    
    Returns:
        LanguageModel: Configured for OpenAI embeddings
    
    Example:
        ```python
        model = openai_embedding("text-embedding-3-small")
        embeddings = await embed(model, "Hello world")
        ```
    """
    client = OpenAI.AsyncOpenAI()
    return LanguageModel(provider="openai", model=model, client=client, options=kwargs)


def google_embedding(model: str = "text-embedding-004", **kwargs: Any) -> LanguageModel:
    """
    Create a LanguageModel instance configured for Google embedding models.
    
    Args:
        model: The Google embedding model name
        **kwargs: Additional options
    
    Returns:
        LanguageModel: Configured for Google embeddings
    
    Example:
        ```python
        model = google_embedding("text-embedding-004")
        embeddings = await embed(model, "Hello world")
        ```
    """
    client = genai.Client()
    return LanguageModel(provider="google", model=model, client=client, options=kwargs)