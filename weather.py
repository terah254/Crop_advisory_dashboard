import httpx
import os

BASE_URL = "https://api.weather-ai.co"

def get_headers():
    return {"Authorization": f"Bearer {os.getenv('WEATHER_AI_KEY')}"}

async def fetch_forecast(lat: float, lon: float, days: int = 7) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/v1/daily",
            headers=get_headers(),
            params={
                "lat": lat,
                "lon": lon,
                "days": days,
                "ai": "false",
                "units": "metric"
            }
        )
        response.raise_for_status()
        return response.json()

async def fetch_current(lat: float, lon: float) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/v1/current",
            headers=get_headers(),
            params={
                "lat": lat,
                "lon": lon,
                "ai": "true",
                "units": "metric"
            }
        )
        response.raise_for_status()
        return response.json()