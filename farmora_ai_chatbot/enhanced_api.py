import os
import uuid
import base64
import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Farmora"
MODEL_NAME = "gemini-2.5-flash"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("farmora")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI(title="Farmora API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LocationInfo(BaseModel):
    latitude: Optional[float] = Field(None)
    longitude: Optional[float] = Field(None)
    town: Optional[str] = None
    province: Optional[str] = None
    country: Optional[str] = None


class ChatRequest(BaseModel):
    message: Optional[str] = ""
    imageBase64: Optional[str] = None
    location: Optional[LocationInfo] = None
    audioBase64: Optional[str] = None
    transcript: Optional[str] = None


class ChatResponse(BaseModel):
    id: str
    response: str
    timestamp: str
    location: str
    hasAudio: bool
    transcript: str


class LocationService:
    @staticmethod
    async def reverse_geocode(lat: float, lon: float) -> str:
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {"lat": lat, "lon": lon, "format": "json"}

            async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": "Farmora/1.0"}) as client:
                res = await client.get(url, params=params)

            if res.status_code == 200:
                data = res.json()
                address = data.get("address", {})
                city = (
                    address.get("city")
                    or address.get("town")
                    or address.get("village")
                    or address.get("hamlet")
                    or ""
                )
                state = address.get("state", "")
                country = address.get("country", "")
                parts = [city, state, country]
                return ", ".join([p for p in parts if p]) or "Unknown location"
        except Exception as e:
            logger.warning(f"Reverse geocoding failed: {e}")

        return "Unknown location"

    @staticmethod
    async def resolve(location: Optional[LocationInfo]) -> str:
        if location:
            if location.latitude and location.longitude:
                return await LocationService.reverse_geocode(
                    location.latitude, location.longitude
                )
            parts = [location.town, location.province, location.country]
            text = ", ".join([p for p in parts if p])
            if text:
                return text
        return "Unknown location"


class AudioService:
    @staticmethod
    def detect(audio_base64: Optional[str]) -> bool:
        if not audio_base64:
            return False
        try:
            audio_bytes = base64.b64decode(audio_base64)
            return len(audio_bytes) > 2000
        except Exception:
            return False


class AIService:
    @staticmethod
    async def generate_text(prompt: str) -> str:
        try:
            response = await asyncio.to_thread(model.generate_content, prompt)
            text = getattr(response, "text", "") or ""
            return text.strip()
        except Exception as e:
            logger.error(f"AI text error: {e}")
            raise

    @staticmethod
    async def analyze_image(image_base64: str, location: str, user_text: str = "") -> str:
        try:
            image_bytes = base64.b64decode(image_base64)

            prompt = f"""
You are an expert agricultural AI assistant.

Farmer location: {location}

Analyze this crop image and give a clean response with:
1. Crop uploaded in the image
2. Confidence score
3. What is wrong with the plant, if anything
4. How to solve the problem
5. How to prevent it in future

Keep it practical, location-aware, and easy to understand.
User message: {user_text}
"""

            response = await asyncio.to_thread(
                model.generate_content,
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
            )

            text = getattr(response, "text", "") or ""
            return text.strip()
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            raise


@app.get("/")
def root():
    return {
        "app": APP_NAME,
        "status": "live",
        "model": MODEL_NAME,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "model": MODEL_NAME,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    request_id = str(uuid.uuid4())

    try:
        location_str = await LocationService.resolve(req.location)
        has_audio = AudioService.detect(req.audioBase64)
        transcript = (req.transcript or "").strip()

        if req.imageBase64:
            result = await AIService.analyze_image(
                req.imageBase64,
                location_str,
                user_text=req.message or transcript,
            )
        else:
            user_input = transcript or req.message or "Hello"

            prompt = f"""
You are a professional agricultural advisor.

Location: {location_str}

Farmer question:
{user_input}

Give a clear, practical, location-aware answer.
"""
            result = await AIService.generate_text(prompt)

        logger.info(f"[{request_id}] Success")

        return ChatResponse(
            id=request_id,
            response=result,
            timestamp=datetime.utcnow().isoformat(),
            location=location_str,
            hasAudio=has_audio,
            transcript=transcript,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "enhanced_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )