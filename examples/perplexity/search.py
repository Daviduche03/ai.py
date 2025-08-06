"""
Web search functionality for Perplexity-like AI responses.
Supports multiple search providers and result processing.
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import quote_plus
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str] = None
    score: float = 0.0


@dataclass
class SearchResponse:
    """Represents the complete search response."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float


class SearchProvider:
    """Base class for search providers."""
    
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        raise NotImplementedError


class SerperSearchProvider(SearchProvider):
    """Google Search via Serper API - most reliable for production."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY environment variable required")
        self.base_url = "https://google.serper.dev/search"
    
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": min(num_results, 20),  # Serper max is 20
            "gl": "us",  # Country
            "hl": "en"   # Language
        }
        
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Serper API error: {response.status}")
                
                data = await response.json()
                search_time = asyncio.get_event_loop().time() - start_time
                
                results = []
                organic_results = data.get("organic", [])
                
                for i, result in enumerate(organic_results[:num_results]):
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        source=self._extract_domain(result.get("link", "")),
                        score=1.0 - (i * 0.1)  # Simple ranking score
                    )
                    results.append(search_result)
                
                return SearchResponse(
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_time=search_time
                )
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except:
            return url


class TavilySearchProvider(SearchProvider):
    """Tavily Search API - designed for AI applications."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable required")
        self.base_url = "https://api.tavily.com/search"
    
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": min(num_results, 20),
            "include_domains": [],
            "exclude_domains": []
        }
        
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Tavily API error: {response.status}")
                
                data = await response.json()
                search_time = asyncio.get_event_loop().time() - start_time
                
                results = []
                tavily_results = data.get("results", [])
                
                for i, result in enumerate(tavily_results):
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        snippet=result.get("content", "")[:300] + "...",  # Truncate content
                        source=self._extract_domain(result.get("url", "")),
                        published_date=result.get("published_date"),
                        score=result.get("score", 1.0 - (i * 0.1))
                    )
                    results.append(search_result)
                
                return SearchResponse(
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_time=search_time
                )
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except:
            return url


class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo search - free but limited."""
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
    
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        # Note: DDG's API is very limited, this is a basic implementation
        # For production, use Serper or Tavily
        
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"DuckDuckGo API error: {response.status}")
                
                data = await response.json()
                search_time = asyncio.get_event_loop().time() - start_time
                
                results = []
                related_topics = data.get("RelatedTopics", [])
                
                for i, topic in enumerate(related_topics[:num_results]):
                    if isinstance(topic, dict) and "Text" in topic:
                        search_result = SearchResult(
                            title=topic.get("Text", "")[:100],
                            url=topic.get("FirstURL", ""),
                            snippet=topic.get("Text", ""),
                            source=self._extract_domain(topic.get("FirstURL", "")),
                            score=1.0 - (i * 0.1)
                        )
                        results.append(search_result)
                
                return SearchResponse(
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_time=search_time
                )
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except:
            return url


class SearchManager:
    """Manages multiple search providers with fallback."""
    
    def __init__(self, providers: List[SearchProvider] = None):
        if providers is None:
            providers = []
            
            # Try to initialize providers based on available API keys
            if os.getenv("SERPER_API_KEY"):
                try:
                    providers.append(SerperSearchProvider())
                    logger.info("Initialized Serper search provider")
                except Exception as e:
                    logger.warning(f"Failed to initialize Serper: {e}")
            
            if os.getenv("TAVILY_API_KEY"):
                try:
                    providers.append(TavilySearchProvider())
                    logger.info("Initialized Tavily search provider")
                except Exception as e:
                    logger.warning(f"Failed to initialize Tavily: {e}")
            
            # Always add DuckDuckGo as fallback (free but limited)
            providers.append(DuckDuckGoSearchProvider())
            logger.info("Initialized DuckDuckGo search provider as fallback")
        
        if not providers:
            raise ValueError("No search providers available. Set SERPER_API_KEY or TAVILY_API_KEY")
        
        self.providers = providers
    
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        """Search using the first available provider with fallback."""
        last_error = None
        
        for provider in self.providers:
            try:
                logger.info(f"Searching with {provider.__class__.__name__}")
                return await provider.search(query, num_results)
            except Exception as e:
                logger.warning(f"Search failed with {provider.__class__.__name__}: {e}")
                last_error = e
                continue
        
        raise Exception(f"All search providers failed. Last error: {last_error}")


# Convenience function for easy usage
async def search_web(query: str, num_results: int = 10, provider: str = "auto") -> SearchResponse:
    """
    Perform web search with automatic provider selection.
    
    Args:
        query: Search query
        num_results: Number of results to return
        provider: Provider to use ("auto", "serper", "tavily", "duckduckgo")
    
    Returns:
        SearchResponse with results
    """
    if provider == "auto":
        manager = SearchManager()
        return await manager.search(query, num_results)
    elif provider == "serper":
        provider_instance = SerperSearchProvider()
        return await provider_instance.search(query, num_results)
    elif provider == "tavily":
        provider_instance = TavilySearchProvider()
        return await provider_instance.search(query, num_results)
    elif provider == "duckduckgo":
        provider_instance = DuckDuckGoSearchProvider()
        return await provider_instance.search(query, num_results)
    else:
        raise ValueError(f"Unknown provider: {provider}")