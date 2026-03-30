# Farmora AI - Scalable API (10K Users Ready)

import os
import uuid
import base64
import asyncio
import logging
import hashlib
from datetime import datetime
from typing import Optional

import httpx
import redis.asyncio as redis
from aiolimiter import AsyncLimiter

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from dotenv import load_dotenv

# ================= CONFIG =================
load_dotenv()

APP_NAME = "Farmora"
MODEL_NAME = "gemini-2.5-flash"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")  # Render Redis

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Redis (cache)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Rate limiter (100 requests per minute per instance)
limiter = AsyncLimiter(100, 60)

# Shared HTTP client
http_client = httpx.AsyncClient(timeout=5.0)

# ================= APP =================
app = FastAPI(title="Farmora API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= MODELS =================
class LocationInfo(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ChatRequest(BaseModel):
    message: Optional[str] = ""
    imageBase64: Optional[str] = None
    location: Optional[LocationInfo] = None
    audioBase64: Optional[str] = None


# ================= HELPERS =================
def hash_input(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def get_cached(key: str):
    return await redis_client.get(key)


async def set_cache(key: str, value: str, ttl=3600):
    await redis_client.set(key, value, ex=ttl)


def detect_audio(audio_base64):
    if not audio_base64:
        return False
    try:
        return len(base64.b64decode(audio_base64)) > 2000
    except:
        return False


async def resolve_location(location: Optional[LocationInfo]):
    if location and location.latitude and location.longitude:
        try:
            res = await http_client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": location.latitude,
                    "lon": location.longitude,
                    "format": "json"
                }
            )
            if res.status_code == 200:
                data = res.json()
                return data.get("display_name", "Unknown location")
        except:
            pass
    return "Unknown location"


# ================= AI =================
async def generate_text(prompt: str):
    return await asyncio.to_thread(model.generate_content, prompt)


async def analyze_image(image_base64, location):
    image_bytes = base64.b64decode(image_base64)

    prompt = f"""
    Farmer location: {location}

    Analyze crop:
    - Plant
    - Health
    - Issue
    - Fix
    """

    return await asyncio.to_thread(
        model.generate_content,
        [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
    )


# ================= ROUTE =================
@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    request_id = str(uuid.uuid4())

    async with limiter:  # rate limit
        try:
            location = await resolve_location(req.location)
            has_audio = detect_audio(req.audioBase64)

            # 🔥 CACHE KEY
            cache_key = hash_input(
                (req.message or "") + (req.imageBase64 or "")
            )

            cached = await get_cached(cache_key)
            if cached:
                return {
                    "id": request_id,
                    "response": cached,
                    "cached": True
                }

            # AI
            if req.imageBase64:
                ai_res = await analyze_image(req.imageBase64, location)
            else:
                ai_res = await generate_text(
                    f"Location: {location}\nQuestion: {req.message}"
                )

            result = ai_res.text.strip()

            if not has_audio:
                result += "\n\n(No audio detected)"

            # 🔥 STORE CACHE
            await set_cache(cache_key, result)

            return {
                "id": request_id,
                "response": result,
                "cached": False,
                "location": location
            }

        except Exception as e:
            logging.error(f"[{request_id}] {e}")
            raise HTTPException(status_code=500, detail="Server error")