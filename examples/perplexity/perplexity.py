"""
Perplexity-like AI search and response generation.
Combines web search with AI to provide cited, accurate answers.
"""

import sys
import os

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from search import SearchManager, SearchResult, SearchResponse, search_web
from ai.core import generateText, streamText
from ai.model import LanguageModel
from ai.tools import Tool
from ai.types import OnFinishResult

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a citation in the response."""
    id: int
    title: str
    url: str
    source: str
    snippet: str


@dataclass
class PerplexityResponse:
    """Complete Perplexity-style response."""
    query: str
    answer: str
    citations: List[Citation]
    search_results: List[SearchResult]
    search_time: float
    generation_time: float
    total_time: float
    conversation_id: Optional[str] = None
    message_count: int = 1


@dataclass
class ConversationMessage:
    """Represents a message in the conversation history."""
    role: str  # 'user' or 'assistant'
    content: str
    citations: Optional[List[Citation]] = None
    search_results: Optional[List[SearchResult]] = None
    timestamp: Optional[float] = None


class SearchParams(BaseModel):
    """Parameters for search tool."""
    query: str = Field(..., description="Search query to find relevant information")


class StockParams(BaseModel):
    """Parameters for stock data tool."""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, GOOGL, TSLA)")
    period: str = Field("1mo", description="Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
    interval: str = Field("1d", description="Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")


class PerplexityEngine:
    """Main engine for Perplexity-like search and response generation."""
    
    def __init__(
        self,
        model: LanguageModel,
        search_manager: SearchManager = None,
        max_search_results: int = 10,
        max_citations: int = 5
    ):
        self.model = model
        self.search_manager = search_manager or SearchManager()
        self.max_search_results = max_search_results
        self.max_citations = max_citations
        
      
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        
        self._current_search_results = []
        self._current_search_time = 0.0
        
        self.search_tool = Tool(
            name="search_web",
            description="Search the web for current information about any topic",
            parameters=SearchParams,
            execute=self._search_execute
        )
        
        # Create stock data tool
        self.stock_tool = Tool(
            name="get_stock_data",
            description="Get stock price data and financial information for charting and analysis",
            parameters=StockParams,
            execute=self._stock_execute
        )
    
    async def _search_execute(self, params: SearchParams) -> Dict[str, Any]:
        """Execute search tool for AI."""
        try:
            search_response = await self.search_manager.search(
                params.query, 
                self.max_search_results
            )
            
            # Format results for AI consumption
            formatted_results = []
            for i, result in enumerate(search_response.results):
                formatted_results.append({
                    "id": i + 1,
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "source": result.source,
                    "score": result.score
                })
            
            return {
                "query": search_response.query,
                "results": formatted_results,
                "total_results": search_response.total_results,
                "search_time": search_response.search_time
            }
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return {"error": str(e), "results": []}

    async def _stock_execute(self, params: StockParams) -> Dict[str, Any]:
        """Execute stock data tool for AI."""
        try:
            import yfinance as yf
            import pandas as pd
            from datetime import datetime, timedelta
            
            logger.info(f"Fetching stock data for {params.symbol}, period={params.period}, interval={params.interval}")
            
            # Create ticker object
            ticker = yf.Ticker(params.symbol)
            
            # Get historical data
            hist = ticker.history(period=params.period, interval=params.interval)
            
            if hist.empty:
                return {
                    "error": f"No data found for symbol {params.symbol}",
                    "symbol": params.symbol
                }
            
            # Get basic info
            try:
                info = ticker.info
                company_name = info.get('longName', params.symbol)
                current_price = info.get('currentPrice', hist['Close'].iloc[-1])
                market_cap = info.get('marketCap', 'N/A')
                pe_ratio = info.get('trailingPE', 'N/A')
                dividend_yield = info.get('dividendYield', 'N/A')
            except:
                company_name = params.symbol
                current_price = hist['Close'].iloc[-1]
                market_cap = 'N/A'
                pe_ratio = 'N/A'
                dividend_yield = 'N/A'
            
            # Format data for charting
            chart_data = []
            for index, row in hist.iterrows():
                chart_data.append({
                    "date": index.strftime('%Y-%m-%d'),
                    "timestamp": int(index.timestamp() * 1000),  # JavaScript timestamp
                    "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2),
                    "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2),
                    "volume": int(row['Volume']) if pd.notna(row['Volume']) else 0
                })
            
            # Calculate basic statistics
            latest_close = float(hist['Close'].iloc[-1])
            previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else latest_close
            change = latest_close - previous_close
            change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
            
            # Calculate moving averages if enough data
            ma_20 = None
            ma_50 = None
            if len(hist) >= 20:
                ma_20 = round(float(hist['Close'].rolling(20).mean().iloc[-1]), 2)
            if len(hist) >= 50:
                ma_50 = round(float(hist['Close'].rolling(50).mean().iloc[-1]), 2)
            
            return {
                "symbol": params.symbol.upper(),
                "company_name": company_name,
                "period": params.period,
                "interval": params.interval,
                "current_price": round(float(current_price), 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "dividend_yield": dividend_yield,
                "ma_20": ma_20,
                "ma_50": ma_50,
                "data_points": len(chart_data),
                "chart_data": chart_data,
                "chart_type": "candlestick",  # Suggest chart type for frontend
                "last_updated": datetime.now().isoformat(),
                "data_range": {
                    "start": chart_data[0]["date"] if chart_data else None,
                    "end": chart_data[-1]["date"] if chart_data else None
                }
            }
            
        except ImportError:
            return {
                "error": "yfinance library not installed. Install with: pip install yfinance",
                "symbol": params.symbol
            }
        except Exception as e:
            logger.error(f"Stock data fetch failed: {e}")
            return {
                "error": f"Failed to fetch stock data: {str(e)}",
                "symbol": params.symbol
            }
    
    def _create_system_message(self, has_conversation_history: bool = False) -> str:
        """Create system message for AI-first approach with tool calling."""
        base_message = """You are Questily, a helpful research assistant.

For this query, decide which tool to use or respond directly:
- Greetings ("hi", "hello"): Respond directly
- Personal questions ("what are you"): Respond directly  
- Current events or research questions: Use the search_web tool
- Stock prices, financial data, charts: Use the get_stock_data tool
- Math or basic facts: Respond directly

If you search, cite sources with [1], [2], [3] format.
If you get stock data, mention the data can be visualized as a chart.
Always provide a helpful response."""

        if has_conversation_history:
            base_message += "\n\nYou have conversation history available. Reference it when relevant."

        return base_message
    
    def get_conversation_history(self, conversation_id: str) -> List[ConversationMessage]:
        """Get conversation history for a given conversation ID."""
        return self.conversations.get(conversation_id, [])
    
    def add_to_conversation(self, conversation_id: str, message: ConversationMessage):
        """Add a message to the conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(message)
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a given conversation ID."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def _build_messages_with_history(self, query: str, conversation_id: str = None) -> List[Dict[str, str]]:
        """Build message history for the AI model."""
        messages = []
        
        if conversation_id:
            history = self.get_conversation_history(conversation_id)
            logger.info(f"Building messages for conversation {conversation_id}: found {len(history)} history messages")
            
            # Add conversation history
            for msg in history:
                if msg.role == "user":
                    messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    # Include the answer with citation info
                    content = msg.content
                    if msg.citations:
                        content += f"\n\n[Previous search sources: {', '.join([c.source for c in msg.citations])}]"
                    messages.append({"role": "assistant", "content": content})
        else:
            logger.info("No conversation_id provided, starting fresh conversation")
        
        messages.append({"role": "user", "content": query})
        logger.info(f"Final messages for AI: {len(messages)} messages")
        
        return messages

    async def stream_search_and_answer(self, query: str, conversation_id: str = None) -> AsyncGenerator[str, None]:
        """Stream a Perplexity-style response - simplified to pass through AI SDK events."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            messages = self._build_messages_with_history(query, conversation_id)
            has_history = len(messages) > 1
            
            if has_history:
                history = self.get_conversation_history(conversation_id)
            
            full_response = ""
            tool_calls_made = []
            tool_results = []
            
            async def callback(result: OnFinishResult):
                print("result from ai", result)
                nonlocal full_response, tool_results, tool_calls_made
                full_response = result["text"]
                tool_results= result['toolResults']
                tool_calls_made = result["toolCalls"]


            async for chunk in streamText(
                model=self.model,
                systemMessage=self._create_system_message(has_history),
                messages=messages,
                tools=[self.search_tool, self.stock_tool],
                onFinish=callback
            ):
                yield chunk
         
            if conversation_id and full_response:
                
                # Add user message
                self.add_to_conversation(conversation_id, ConversationMessage(
                    role="user",
                    content=query,
                    timestamp=start_time
                ))
                
                self.add_to_conversation(conversation_id, ConversationMessage(
                    role="assistant",
                    content=full_response,
                    citations=None,
                    search_results=None, 
                    timestamp=asyncio.get_event_loop().time()
                ))
                
              
                current_history = self.get_conversation_history(conversation_id)
            elif not conversation_id:
                logger.warning("No conversation_id provided, not storing history")
            elif not full_response:
                logger.warning("No response generated, not storing history")
            
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    def _format_search_results_for_ai(self, results: List[SearchResult]) -> str:
        """Format search results for AI consumption."""
        formatted = []
        for i, result in enumerate(results[:self.max_citations]):
            formatted.append(f"[{i+1}] {result.title}\nSource: {result.source}\nURL: {result.url}\nContent: {result.snippet}\n")
        return "\n".join(formatted)



_global_engine = None

def get_global_engine(model: LanguageModel, max_results: int = 10) -> PerplexityEngine:
    """Get or create a global engine instance for conversation persistence."""
    global _global_engine
    if _global_engine is None:
        search_manager = SearchManager()
        _global_engine = PerplexityEngine(model, search_manager, max_results)
        logger.info("Created new global PerplexityEngine")
    elif _global_engine.model != model:
        # Update model but keep conversation history
        logger.info(f"Updating global engine model from {_global_engine.model.model} to {model.model}")
        _global_engine.model = model
    
    return _global_engine


async def perplexity_stream(
    query: str,
    model: LanguageModel,
    search_provider: str = "auto",
    max_results: int = 10,
    conversation_id: str = None
) -> AsyncGenerator[str, None]:
    """
    Stream a Perplexity-style search and answer.
    
    Args:
        query: The search query
        model: Language model to use for response generation
        search_provider: Search provider ("auto", "serper", "tavily", "duckduckgo")
        max_results: Maximum number of search results
        conversation_id: Optional conversation ID for history tracking
    
    Yields:
        Server-sent events with search results and streaming answer
    """
    engine = get_global_engine(model, max_results)
    
    async for chunk in engine.stream_search_and_answer(query, conversation_id):
        yield chunk


def clear_conversation_history(conversation_id: str):
    """Clear conversation history for a given conversation ID."""
    global _global_engine
    if _global_engine:
        _global_engine.clear_conversation(conversation_id)


def get_conversation_history(conversation_id: str) -> List[ConversationMessage]:
    """Get conversation history for a given conversation ID."""
    global _global_engine
    if _global_engine:
        return _global_engine.get_conversation_history(conversation_id)
    return []