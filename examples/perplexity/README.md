# Perplexity Clone

A complete Perplexity-like AI search engine built with the AI SDK. Combines real-time web search with AI to provide accurate, cited answers.

## Features

- **Real-time web search** - Multiple search providers (Serper, Tavily, DuckDuckGo)
- **AI-powered answers** - Uses GPT-4, Gemini, and other models
- **Source citations** - Automatic citation generation with links
- **Stock data & charts** - Real-time stock prices with interactive charts
- **Streaming responses** - Real-time answer generation
- **Web interface** - Beautiful, responsive frontend
- **REST API** - Complete API for integration

## Quick Start

### 1. Setup

```bash
cd examples/perplexity
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Required
export OPENAI_API_KEY="your-openai-key"

# Optional (choose one or more for better search)
export SERPER_API_KEY="your-serper-key"      # Google Search (recommended)
export TAVILY_API_KEY="your-tavily-key"      # AI-optimized search

# Stock data is free via yfinance (no API key needed)
```

### 3. Run the Server

```bash
python app.py
```

### 4. Open in Browser

Visit `http://localhost:8000` to use the web interface.

## API Keys Setup

### OpenAI (Required)
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Set `OPENAI_API_KEY` environment variable

### Serper - Google Search (Recommended)
1. Go to [Serper.dev](https://serper.dev/)
2. Sign up and get your API key
3. Set `SERPER_API_KEY` environment variable

### Tavily - AI Search (Alternative)
1. Go to [Tavily.com](https://tavily.com/)
2. Sign up and get your API key  
3. Set `TAVILY_API_KEY` environment variable

## API Usage

### Basic Search

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest AI developments in 2024?",
    "model": "gpt-4"
  }'
```

### Python SDK Usage

```python
import sys
sys.path.append('../..')  # Add main AI SDK to path

from ai.model import openai
from search import search_web
from perplexity import perplexity_search

# Perform search and get answer
result = await perplexity_search(
    query="What is the latest news about AI?",
    model=openai("gpt-4")
)

print(f"Answer: {result.answer}")
print(f"Citations: {len(result.citations)}")
```

## How It Works

1. **Query Processing**: User submits a search query
2. **AI Decision**: AI decides whether to search web, get stock data, or respond directly
3. **Tool Execution**: System executes appropriate tools (web search, stock API)
4. **Data Processing**: Results are processed and formatted for display
5. **Response Generation**: AI generates comprehensive answer with citations/charts
6. **Streaming Delivery**: Response is streamed to user in real-time

## Stock Features

- **Real-time stock prices** - Current price, change, percentage
- **Interactive charts** - Candlestick and line charts with Chart.js
- **Technical indicators** - Moving averages (MA20, MA50)
- **Company info** - Market cap, P/E ratio, dividend yield
- **Flexible periods** - 1d, 1mo, 3mo, 1y, 2y, 5y, 10y, max
- **Multiple intervals** - 1m, 5m, 1h, 1d, 1wk, 1mo

### Stock Query Examples:
- "Show me AAPL stock chart for the last 6 months"
- "What's Tesla's stock performance this year?"
- "Compare GOOGL vs MSFT stock prices"
- "NVDA stock analysis with technical indicators"

## Troubleshooting

**"No search providers available"**
- Set at least one search API key (SERPER_API_KEY or TAVILY_API_KEY)
- Or rely on DuckDuckGo fallback (limited functionality)

**"OpenAI API key required"**
- Set OPENAI_API_KEY environment variable

**Slow responses**
- Use faster models (gpt-3.5-turbo, gemini-1.5-flash)
- Reduce max_results parameter
- Use streaming for better perceived performance