import httpx
import os

BASE_URL = "https://api.weather-ai.co"

def get_headers():
    return {"Authorization": f"Bearer {os.getenv('WEATHER_AI_KEY')}"}

async def analyze_farm_image(
    image_bytes: bytes,
    filename: str,
    county: str = "",
    land_acres: float = None
) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        files = {"image": (filename, image_bytes, "image/jpeg")}
        data  = {"county": county}
        if land_acres:
            data["landAcres"] = str(land_acres)

        response = await client.post(
            f"{BASE_URL}/v1/trees/analyze",
            headers=get_headers(),
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json()