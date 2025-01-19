# AI Movie Recommendation Agent

This is a demo application showcasing the power of pydanticAI for building intelligent agents. The application provides personalized movie recommendations based on user preferences, leveraging the TMDB API for movie data and OpenAI's GPT-4 for intelligent recommendation generation.

## Features

- Modern web interface built with FastAPI and Tailwind CSS
- Integration with TMDB API for real movie data
- Intelligent movie recommendations using GPT-4
- Structured responses using Pydantic models
- Asynchronous API calls for better performance

## Setup

1. Install Poetry (if you haven't already):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Create a `.env` file in the project root with your API keys:
```
TMDB_API_KEY=your_tmdb_api_key
OPENAI_API_KEY=your_openai_api_key
```

4. Run the application:
```bash
poetry run python app.py
```

5. Open your browser and navigate to `http://localhost:8000`

## How it Works

1. The user enters their movie preferences in the web interface
2. The pydanticAI agent processes the preferences using GPT-4
3. The agent uses TMDB API to search for and get details about relevant movies
4. The agent generates personalized recommendations with explanations
5. The results are displayed in a user-friendly format

## Dependencies

- pydantic-ai: For building the intelligent agent
- fastapi: Web framework
- uvicorn: ASGI server
- jinja2: Template engine
- python-dotenv: Environment variable management
- httpx: Async HTTP client
