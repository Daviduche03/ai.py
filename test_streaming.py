import asyncio
import aiohttp

async def test_streaming():
    """Test the new streaming format with tool calls"""
    url = "http://localhost:8001/api/chat"
    
    # Test message that should trigger a tool call
    payload = {
        "messages": [
            {"role": "user", "content": "What's the weather like in Tokyo?"}
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            print(f"Response status: {response.status}")
            print("Streaming response:")
            print("-" * 50)
            
            async for line in response.content:
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line:
                        print(decoded_line)

if __name__ == "__main__":
    asyncio.run(test_streaming())