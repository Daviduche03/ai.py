"""
Perplexity Clone - FastAPI Backend
A complete Perplexity-like search and AI response system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import logging
import asyncio

from ai.model import openai, google
from perplexity import (
    PerplexityEngine, perplexity_stream,
    clear_conversation_history, get_conversation_history
)
from search import SearchManager, SerperSearchProvider, TavilySearchProvider
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Perplexity Clone API",
    description="AI-powered search with citations and real-time responses",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    model: Optional[str] = Field("gpt-4", description="AI model to use")
    search_provider: Optional[str] = Field("auto", description="Search provider")
    max_results: Optional[int] = Field(10, description="Maximum search results", ge=1, le=20)
    stream: Optional[bool] = Field(False, description="Stream response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for history tracking")


class ConversationRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation ID to manage")


# Global search manager (initialized on startup)
search_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize search providers on startup."""
    global search_manager
    
    logger.info("Initializing Perplexity Clone API...")
    
    # Check for required API keys
    required_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        required_keys.append("OPENAI_API_KEY")
    
    search_keys = []
    if not os.getenv("SERPER_API_KEY"):
        search_keys.append("SERPER_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        search_keys.append("TAVILY_API_KEY")
    
    if required_keys:
        logger.error(f"Missing required environment variables: {', '.join(required_keys)}")
        raise RuntimeError(f"Missing API keys: {', '.join(required_keys)}")
    
    if search_keys:
        logger.warning(f"Missing search API keys: {', '.join(search_keys)}. Using DuckDuckGo fallback.")
    
    # Initialize search manager
    try:
        search_manager = SearchManager()
        logger.info("Search manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize search manager: {e}")
        raise RuntimeError(f"Search initialization failed: {e}")
    
    logger.info("✅ Perplexity Clone API ready!")


@app.get("/")
async def root():
    """Serve the main frontend."""
    return FileResponse("static/index.html")


@app.get("/api")
async def api_info():
    """API health check and information."""
    return {
        "name": "Perplexity Clone API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "stream": "/api/search/stream",
            "health": "/health"
        },
        "docs": "/docs"
    }
    

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "search_providers": len(search_manager.providers) if search_manager else 0,
        "timestamp": asyncio.get_event_loop().time()
    }


@app.post("/api/search/stream")
async def stream_search_endpoint(request: SearchRequest):
    """
    Stream AI-powered search with real-time results.
    Simplified to pass through AI SDK events directly to frontend.
    """
    try:
        # Select AI model
        if request.model.startswith("gpt"):
            model = openai(request.model)
        elif request.model.startswith("gemini"):
            model = google(request.model)
        else:
            model = openai("gpt-4")  # Default fallback
        
        async def generate_stream():
            try:
                async for chunk in perplexity_stream(
                    query=request.query,
                    model=model,
                    search_provider=request.search_provider,
                    max_results=request.max_results,
                    conversation_id=request.conversation_id
                ):
                    yield chunk
                
            except Exception as e:
                logger.error(f"Streaming failed: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Stream setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversation/new")
async def create_conversation():
    """Create a new conversation and return its ID."""
    conversation_id = str(uuid.uuid4())
    return {
        "conversation_id": conversation_id,
        "created_at": asyncio.get_event_loop().time(),
        "message_count": 0
    }


@app.get("/api/conversation/{conversation_id}/history")
async def get_conversation_history_endpoint(conversation_id: str):
    """Get conversation history for a given conversation ID."""
    try:
        history = get_conversation_history(conversation_id)
        
        # Convert to API format
        formatted_history = []
        for msg in history:
            formatted_msg = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            
            if msg.citations:
                formatted_msg["citations"] = [
                    {
                        "id": c.id,
                        "title": c.title,
                        "url": c.url,
                        "source": c.source,
                        "snippet": c.snippet
                    }
                    for c in msg.citations
                ]
            
            if msg.search_results:
                formatted_msg["search_results_count"] = len(msg.search_results)
            
            formatted_history.append(formatted_msg)
        
        return {
            "conversation_id": conversation_id,
            "history": formatted_history,
            "message_count": len(formatted_history),
            "last_updated": formatted_history[-1]["timestamp"] if formatted_history else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversation/{conversation_id}")
async def clear_conversation_endpoint(conversation_id: str):
    """Clear conversation history for a given conversation ID."""
    try:
        clear_conversation_history(conversation_id)
        return {
            "conversation_id": conversation_id,
            "status": "cleared",
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversation/{conversation_id}/summary")
async def get_conversation_summary(conversation_id: str):
    """Get a summary of the conversation."""
    try:
        history = get_conversation_history(conversation_id)
        
        if not history:
            return {
                "conversation_id": conversation_id,
                "exists": False,
                "message_count": 0
            }
        
        user_messages = [msg for msg in history if msg.role == "user"]
        assistant_messages = [msg for msg in history if msg.role == "assistant"]
        
        # Get unique sources from all citations
        all_sources = set()
        for msg in assistant_messages:
            if msg.citations:
                for citation in msg.citations:
                    all_sources.add(citation.source)
        
        return {
            "conversation_id": conversation_id,
            "exists": True,
            "message_count": len(history),
            "user_queries": len(user_messages),
            "ai_responses": len(assistant_messages),
            "unique_sources": len(all_sources),
            "sources": list(all_sources),
            "first_query": user_messages[0].content if user_messages else None,
            "last_query": user_messages[-1].content if user_messages else None,
            "created_at": history[0].timestamp if history else None,
            "last_updated": history[-1].timestamp if history else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        exit(1)
    
    # Warn about search API keys
    if not os.getenv("SERPER_API_KEY") and not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Warning: No search API keys found")
        print("For better search results, set one of:")
        print("  - SERPER_API_KEY (recommended)")
        print("  - TAVILY_API_KEY")
        print("Using DuckDuckGo fallback (limited functionality)")
    
    print("🚀 Starting Perplexity Clone API...")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )