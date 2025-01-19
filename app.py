from dataclasses import dataclass
import os
from typing import List, Optional
import logging
from dotenv import load_dotenv

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

@dataclass
class MovieDependencies:
    tmdb_api_key: str
    image_base_url: str = TMDB_IMAGE_BASE_URL

class MovieRecommendation(BaseModel):
    id: int = Field(description="Movie ID")
    title: str = Field(description="Movie title")
    description: str = Field(description="Movie description")
    poster_path: Optional[str] = Field(description="Path to movie poster image")
    backdrop_path: Optional[str] = Field(description="Path to movie backdrop image")
    rating: float = Field(description="Movie rating out of 10", default=0.0)

class MovieRecommendations(BaseModel):
    recommendations: List[MovieRecommendation] = Field(description="List of movie recommendations")
    explanation: str = Field(description="Brief explanation of the recommendations")
    image_base_url: str = Field(description="Base URL for movie images", default="https://image.tmdb.org/t/p/w500")
    tmdb_base_url: str = Field(description="Base URL for TMDB movie pages", default="https://www.themoviedb.org/movie/")

class PreferenceRequest(BaseModel):
    preferences: str

movie_agent = Agent(
    "openai:gpt-4",
    deps_type=MovieDependencies,
    result_type=MovieRecommendations,
    system_prompt=(
        "You are a knowledgeable movie recommendation assistant. "
        "Provide 3-5 diverse personalized movie recommendations based on user preferences. "
        "Pay careful attention to specific requirements like time periods, genres, and whether "
        "movies should be live-action or animated. When a user asks for live-action movies, "
        "make sure to set exclude_animation=True in discover_movies. "
        "For Disney movies, use company='Disney' in discover_movies. "
        "When date ranges are specified, always use both year_from and year_to parameters. "
        "Keep explanations concise but informative. "
        "Make sure to include the vote_average as the rating for each movie."
    ),
)

@movie_agent.tool
async def search_movies(ctx: RunContext[MovieDependencies], query: str) -> dict:
    """Search for movies using TMDB API. Returns a list of movies matching the query."""
    async with httpx.AsyncClient() as client:
        logger.info(f"Searching TMDB for query: {query}")
        response = await client.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": ctx.deps.tmdb_api_key,
                "query": query,
                "language": "en-US",
                "page": 1,
                "include_adult": False
            },
        )
        data = response.json()
        # Limit results and only include necessary fields
        results = data.get('results', [])[:5]
        filtered_results = [{
            'id': movie['id'],
            'title': movie['title'],
            'overview': movie.get('overview', '')[:200],  # Limit overview length
            'release_date': movie.get('release_date', ''),
            'poster_path': movie.get('poster_path'),
            'backdrop_path': movie.get('backdrop_path'),
            'vote_average': movie.get('vote_average', 0.0),  # Include rating
        } for movie in results]
        return {'results': filtered_results}

@movie_agent.tool
async def discover_movies(
    ctx: RunContext[MovieDependencies], 
    genre: str = "", 
    year_from: int = None, 
    year_to: int = None,
    exclude_animation: bool = False,
    company: str = None
) -> dict:
    """Discover movies based on various filters.
    
    Args:
        genre: Genre name to filter by
        year_from: Start year for release date range
        year_to: End year for release date range
        exclude_animation: If True, excludes animated movies
        company: Production company name (e.g., 'Disney', 'Warner Bros')
    """
    async with httpx.AsyncClient() as client:
        # Get genre IDs
        genre_response = await client.get(
            "https://api.themoviedb.org/3/genre/movie/list",
            params={
                "api_key": ctx.deps.tmdb_api_key,
                "language": "en-US",
            }
        )
        genres = genre_response.json().get("genres", [])
        
        # Get animation genre ID
        animation_id = next((g["id"] for g in genres if g["name"].lower() == "animation"), None)
        
        # Get specified genre ID
        genre_id = None
        if genre:
            for g in genres:
                if genre.lower() in g["name"].lower():
                    genre_id = g["id"]
                    break

        # Get company ID if specified
        company_id = None
        if company:
            company_response = await client.get(
                "https://api.themoviedb.org/3/search/company",
                params={
                    "api_key": ctx.deps.tmdb_api_key,
                    "query": company,
                }
            )
            companies = company_response.json().get("results", [])
            if companies:
                company_id = companies[0]["id"]

        params = {
            "api_key": ctx.deps.tmdb_api_key,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "include_adult": False,
            "page": 1,
        }
        
        if genre_id:
            params["with_genres"] = genre_id
        if year_from:
            params["primary_release_date.gte"] = f"{year_from}-01-01"
        if year_to:
            params["primary_release_date.lte"] = f"{year_to}-12-31"
        if company_id:
            params["with_companies"] = company_id
        if exclude_animation and animation_id:
            params["without_genres"] = animation_id

        logger.info(f"Discovering movies with params: {params}")
        response = await client.get(
            "https://api.themoviedb.org/3/discover/movie",
            params=params,
        )
        data = response.json()
        results = data.get('results', [])[:5]
        filtered_results = [{
            'id': movie['id'],
            'title': movie['title'],
            'overview': movie.get('overview', '')[:200],
            'release_date': movie.get('release_date', ''),
            'poster_path': movie.get('poster_path'),
            'backdrop_path': movie.get('backdrop_path'),
            'vote_average': movie.get('vote_average', 0.0),
        } for movie in results]
        return {'results': filtered_results}

@movie_agent.tool
async def get_movie_details(ctx: RunContext[MovieDependencies], movie_id: int) -> dict:
    """Get detailed information about a specific movie."""
    async with httpx.AsyncClient() as client:
        logger.info(f"Getting details for movie ID: {movie_id}")
        response = await client.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={
                "api_key": ctx.deps.tmdb_api_key,
                "language": "en-US",
            },
        )
        movie = response.json()
        # Return only necessary fields
        return {
            'id': movie['id'],
            'title': movie['title'],
            'overview': movie.get('overview', '')[:200],  # Limit overview length
            'release_date': movie.get('release_date', ''),
            'poster_path': movie.get('poster_path'),
            'backdrop_path': movie.get('backdrop_path'),
            'vote_average': movie.get('vote_average', 0.0),  # Include rating
            'genres': [g['name'] for g in movie.get('genres', [])],
        }

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend")
async def recommend(request: PreferenceRequest):
    logger.info(f"Received request body: {request.model_dump_json()}")
    deps = MovieDependencies(
        tmdb_api_key=TMDB_API_KEY,
        image_base_url=TMDB_IMAGE_BASE_URL,
    )
    
    result = await movie_agent.run(
        f"Recommend movies based on these preferences: {request.preferences}. "
        "For each movie, include both poster_path and backdrop_path from the TMDB API response. "
        "These paths should start with a forward slash.",
        deps=deps
    )
    result.data.image_base_url = TMDB_IMAGE_BASE_URL
    result.data.tmdb_base_url = "https://www.themoviedb.org/movie/"
    return result.data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
