from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ai.core import streamText, generateText
from ai.model import openai, google
from ai.tools import Tool
from ai.types import OnFinishResult
from pydantic import BaseModel, Field
import random
import logging
import json

# It is recommended to set the OpenAI API key as an environment variable.
# os.environ["OPENAI_API_KEY"] = "your-api-key"

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

    class WeatherParams(BaseModel):
        location: str = Field(..., description="City and country e.g. Bogotá, Colombia")

    weather_tool = Tool(
        name="get_weather",
        description="Get current temperature for a given location.",
        parameters=WeatherParams,
        execute=lambda params: {"location": params.location, "temperature": random.randint(-20, 35)},
    )

    stream = streamText(
        model=google("gemini-2.5-pro"),
        messages=messages,
        systemMessage="Your name is david.",
        tools=[weather_tool],
        onFinish=on_finish_callback,
        # options=body,
    )

    async def generate():
        async for chunk in stream:
            yield f"data: {chunk}\n\n"

    return StreamingResponse(generate(), media_type="text/plain")


# Example onFinish callback for logging and analytics
async def on_finish_callback(result: OnFinishResult):
    """Example callback that logs completion details and could send to analytics."""
    logging.info(f"Generation completed:")
    logging.info(f"  - Finish reason: {result['finishReason']}")
    logging.info(f"  - Token usage: {result['usage']}")
    logging.info(f"  - Text length: {len(result['text'])} characters")
    logging.info(f"  - Tool calls made: {len(result['toolCalls'])}")
    logging.info(f"  - Tool results: {len(result['toolResults'])}")
    
    # Here you could send data to analytics, save to database, etc.
    # Example: await send_to_analytics(result)
    # Example: await save_conversation_log(result)

@app.post("/api/chat-non-streaming")
async def chat_non_streaming(request: Request):
    body = await request.json()
    messages = body.pop("messages", [])

    if not messages:
        return {"error": "messages is required"}, 400

    class WeatherParams(BaseModel):
        location: str = Field(..., description="City and country e.g. Bogotá, Colombia")

    weather_tool = Tool(
        name="get_weather",
        description="Get current temperature for a given location.",
        parameters=WeatherParams,
        execute=lambda params: {"location": params.location, "temperature": random.randint(-20, 35)},
    )

    response = await generateText(
        model=google("gemini-2.5-pro"),
        messages=messages,
        systemMessage="Your name is david.",
        tools=[weather_tool],
        onFinish=on_finish_callback,  # Add the callback
        # options=body,
    )

    return {"response": response}
