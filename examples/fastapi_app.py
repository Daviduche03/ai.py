import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ai.core import streamText, generateText, embed, embedMany
from ai.model import openai, google, openai_embedding
from ai.image import create_image_message, image_from_url, image_from_base64
from ai.tools import Tool
from ai.types import OnFinishResult
from pydantic import BaseModel, Field
import random
import logging
import json


# os.environ["OPENAI_API_KEY"] = "your-api-key"

logging.basicConfig(level=logging.INFO)

app = FastAPI()


class WeatherParams(BaseModel):
    location: str = Field(..., description="City and country e.g. Bogotá, Colombia")

class CallNumberParams(BaseModel):
    number: str = Field(..., description="Phone number to call")

weather_tool = Tool(
    name="get_weather",
    description="Get current temperature for a given location.",
    parameters=WeatherParams,
    execute=lambda params: {"location": params.location, "temperature": random.randint(-20, 35)},
)

# Client-side tool (no execute function)
call_number_tool = Tool(
    name="callMyNumber",
    description="Call a phone number (client-side execution)",
    parameters=CallNumberParams,
    # No execute function - this will be handled client-side
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.pop("messages", [])

    if not messages:
        return {"error": "messages is required"}, 400

    # stream = streamText(
    #     model=openai('gpt-4.1'),
    #     messages=messages,
    #     systemMessage="Your name is david.",
    #     tools=[weather_tool],
    #     onFinish=on_finish_callback,
    #     # options=body,
    # )

    return StreamingResponse(
         streamText(
            model=openai('gpt-4.1'),
            messages=messages,
            systemMessage="Your name is david.",
            tools=[weather_tool, call_number_tool],
            onFinish=on_finish_callback,
        ),
        media_type="text/plain; charset=utf-8",
    )


async def on_finish_callback(result: OnFinishResult):
    """Example callback that logs completion details and could send to analytics."""
    logging.info(f"Generation completed:")
    logging.info(f"  - Finish reason: {result['finishReason']}")
    logging.info(f"  - Token usage: {result['usage']}")
    logging.info(f"  - Text length: {len(result['text'])} characters")
    logging.info(f"  - Tool calls made: {len(result['toolCalls'])}")
    logging.info(f"  - Tool results: {len(result['toolResults'])}")
    
   

@app.post("/api/chat-non-streaming")
async def chat_non_streaming(request: Request):
    body = await request.json()
    messages = body.pop("messages", [])

    if not messages:
        return {"error": "messages is required"}, 400

    response = await generateText(
        model=google("gemini-2.5-pro"),
        messages=messages,
        systemMessage="Your name is david.",
        tools=[weather_tool],
        onFinish=on_finish_callback,  # Add the callback
        # options=body,
    )

    return {"response": response}


    """Calculate semantic similarity between two texts."""
    body = await request.json()
    text1 = body.get("text1")
    text2 = body.get("text2")
    
    if not text1 or not text2:
        return {"error": "Both text1 and text2 are required"}, 400
    
    try:
        model = openai_embedding("text-embedding-3-small")
        embeddings = await embedMany(model, [text1, text2])
        
        # Calculate cosine similarity
        import math
        
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(x * x for x in b))
            
            if magnitude_a == 0 or magnitude_b == 0:
                return 0
            
            return dot_product / (magnitude_a * magnitude_b)
        
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        
        return {
            "text1": text1,
            "text2": text2,
            "similarity": similarity,
            "interpretation": (
                "Very similar" if similarity > 0.8 else
                "Similar" if similarity > 0.6 else
                "Somewhat similar" if similarity > 0.4 else
                "Different"
            )
        }
    except Exception as e:
        return {"error": str(e)}, 500

    """Streaming chat that supports images."""
    body = await request.json()
    messages = body.get("messages", [])
    
    if not messages:
        return {"error": "messages is required"}, 400
    
    # The messages can already contain images in the proper format
    # No additional processing needed since our validation handles it
    
    return StreamingResponse(
        streamText(
            model=openai("gpt-4o"),  # Vision-capable model
            systemMessage="You are a helpful assistant that can see and analyze images.",
            messages=messages
        ),
        media_type="text/plain; charset=utf-8"
    )