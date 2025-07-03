from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from ai.core import streamText, generateText
from ai.model import openai, google
from ai.tools import Tool
from pydantic import BaseModel, Field
import random
import logging

# It is recommended to set the OpenAI API key as an environment variable.
# os.environ["OPENAI_API_KEY"] = "your-api-key"

logging.basicConfig(level=logging.INFO)

app = FastAPI()


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

    return StreamingResponse(
         streamText(
            model=openai("gpt-4.1"),
            messages=messages,
            systemMessage="Your name is david.",
            tools=[weather_tool],
            # options=body,
        ),
        media_type="text/plain; charset=utf-8",
    )


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
        model=openai("gpt-4.1"),
        messages=messages,
        systemMessage="Your name is david.",
        tools=[weather_tool],
        # options=body,
    )

    return {"response": response}
