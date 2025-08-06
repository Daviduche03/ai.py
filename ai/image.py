"""
Image utilities for the Python AI SDK.

This module provides helper functions for working with images in AI messages,
including base64 encoding and message formatting.
"""

import base64
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Union
import requests
from io import BytesIO


def image_from_file(file_path: str) -> Dict[str, Any]:
    """
    Create an image message part from a local file (Vercel AI SDK format).
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dict containing the image message part
        
    Example:
        ```python
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                image_from_file("path/to/image.jpg")
            ]
        }
        ```
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Read file as bytes
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    
    return {
        "type": "image",
        "image": image_bytes
    }


def image_from_url(url: str) -> Dict[str, Any]:
    """
    Create an image message part from a URL (Vercel AI SDK format).
    
    Args:
        url: URL of the image
        
    Returns:
        Dict containing the image message part
        
    Example:
        ```python
        message = {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Describe this image"},
                image_from_url("https://example.com/image.jpg")
            ]
        }
        ```
    """
    if not url.startswith(("http://", "https://", "data:")):
        raise ValueError("URL must start with http://, https://, or data:")
    
    return {
        "type": "image",
        "image": url
    }


def image_from_base64(base64_data: str) -> Dict[str, Any]:
    """
    Create an image message part from base64 data (Vercel AI SDK format).
    
    Args:
        base64_data: Base64 encoded image data (without data URL prefix)
        
    Returns:
        Dict containing the image message part
        
    Example:
        ```python
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see?"},
                image_from_base64(base64_string)
            ]
        }
        ```
    """
    # Validate base64 data
    try:
        base64.b64decode(base64_data, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {e}")
    
    return {
        "type": "image",
        "image": base64_data
    }


def image_from_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Create an image message part from raw bytes (Vercel AI SDK format).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dict containing the image message part
    """
    return {
        "type": "image",
        "image": image_bytes
    }


def file_from_path(file_path: str, media_type: str = None, filename: str = None) -> Dict[str, Any]:
    """
    Create a file message part from a local file (Vercel AI SDK format).
    
    Args:
        file_path: Path to the file
        media_type: MIME type of the file (auto-detected if not provided)
        filename: Optional filename (defaults to basename of file_path)
        
    Returns:
        Dict containing the file message part
        
    Example:
        ```python
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this PDF?"},
                file_from_path("document.pdf", "application/pdf")
            ]
        }
        ```
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect media type if not provided
    if not media_type:
        media_type, _ = mimetypes.guess_type(file_path)
        if not media_type:
            media_type = "application/octet-stream"
    
    # Use basename as filename if not provided
    if not filename:
        filename = path.name
    
    # Read file as bytes
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    return {
        "type": "file",
        "data": file_bytes,
        "mediaType": media_type,
        "filename": filename
    }


def file_from_bytes(file_bytes: bytes, media_type: str, filename: str = None) -> Dict[str, Any]:
    """
    Create a file message part from raw bytes (Vercel AI SDK format).
    
    Args:
        file_bytes: Raw file bytes
        media_type: MIME type of the file
        filename: Optional filename
        
    Returns:
        Dict containing the file message part
    """
    file_part = {
        "type": "file",
        "data": file_bytes,
        "mediaType": media_type
    }
    
    if filename:
        file_part["filename"] = filename
    
    return file_part


def download_and_encode_image(url: str) -> Dict[str, Any]:
    """
    Download an image from URL and convert to bytes format.
    
    This is useful when you want to ensure the image is embedded in the message
    rather than referenced by URL.
    
    Args:
        url: URL of the image to download
        
    Returns:
        Dict containing the image message part with bytes data
        
    Example:
        ```python
        # Download and embed image
        image_part = download_and_encode_image("https://example.com/image.jpg")
        
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image"},
                image_part
            ]
        }
        ```
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        return image_from_bytes(response.content)
        
    except requests.RequestException as e:
        raise ValueError(f"Failed to download image from {url}: {e}")


def download_file(url: str, media_type: str = None, filename: str = None) -> Dict[str, Any]:
    """
    Download a file from URL and create a file message part.
    
    Args:
        url: URL of the file to download
        media_type: MIME type (auto-detected if not provided)
        filename: Optional filename
        
    Returns:
        Dict containing the file message part
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Auto-detect media type from headers or URL
        if not media_type:
            media_type = response.headers.get("content-type")
            if not media_type:
                media_type, _ = mimetypes.guess_type(url)
                if not media_type:
                    media_type = "application/octet-stream"
        
        # Extract filename from URL if not provided
        if not filename:
            filename = Path(url).name or "downloaded_file"
        
        return file_from_bytes(response.content, media_type, filename)
        
    except requests.RequestException as e:
        raise ValueError(f"Failed to download file from {url}: {e}")


def create_image_message(text: str, *images: Union[str, Dict[str, Any]], role: str = "user") -> Dict[str, Any]:
    """
    Create a complete message with text and images.
    
    Args:
        text: Text content of the message
        *images: Image parts (file paths, URLs, or image dicts)
        role: Message role ("user" or "assistant")
        
    Returns:
        Complete message dict ready for AI API
        
    Example:
        ```python
        # Mix of file paths and URLs
        message = create_image_message(
            "Compare these images",
            "local_image.jpg",
            "https://example.com/remote.png",
            image_from_base64(base64_data)
        )
        ```
    """
    content = [{"type": "text", "text": text}]
    
    for image in images:
        if isinstance(image, str):
            # Assume it's a file path or URL
            if image.startswith(("http://", "https://")):
                content.append(image_from_url(image))
            else:
                content.append(image_from_file(image))
        elif isinstance(image, dict):
            # Assume it's already a formatted image part
            content.append(image)
        else:
            raise ValueError(f"Invalid image type: {type(image)}")
    
    return {
        "role": role,
        "content": content
    }


# Convenience functions for common patterns
def text_with_image(text: str, image_path: str) -> Dict[str, Any]:
    """Quick helper for text + single local image."""
    return create_image_message(text, image_path)


def text_with_url_image(text: str, image_url: str) -> Dict[str, Any]:
    """Quick helper for text + single URL image.""" 
    return create_image_message(text, image_url)