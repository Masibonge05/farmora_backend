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

model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config=genai.types.GenerationConfig(
        temperature=0.2,
        max_output_tokens=1800,
    ),
)

app = FastAPI(title="Farmora API", version="5.0.0")

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
    language: Optional[str] = "en"


class ChatResponse(BaseModel):
    id: str
    response: str
    timestamp: str
    location: str
    hasAudio: bool
    transcript: str
    language: str


LANGUAGE_NAMES = {
    "en": "English",
    "zu": "isiZulu",
    "xh": "isiXhosa",
    "af": "Afrikaans",
    "st": "Sesotho",
}


def get_language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, "English")


class LocationService:
    @staticmethod
    async def reverse_geocode(lat: float, lon: float) -> str:
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {"lat": lat, "lon": lon, "format": "json"}

            async with httpx.AsyncClient(
                timeout=8.0,
                headers={"User-Agent": "Farmora/1.0"}
            ) as client:
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
                    location.latitude,
                    location.longitude,
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


class PromptService:
    @staticmethod
    def image_prompt(location: str, language_name: str, user_text: str) -> str:
        return f"""
You are Farmora AI, an expert agricultural assistant.

Farmer location: {location}
Reply language: {language_name}

The user uploaded a plant or crop image.
If the user also provided a question or transcript, use it as extra context.

Return the answer in clean plain text only.
Do not use markdown.
Do not use asterisks.
Do not use bullet symbols unless necessary.
Use this exact structure with headings:

Crop:
Confidence:
Health Status:
Problem:
Treatment:
Prevention:

Rules:
- Identify the crop if possible.
- Give a realistic confidence score as a percentage.
- If the plant looks healthy, say so clearly in Health Status and Problem.
- Treatment must explain what the farmer should do now.
- Prevention must explain how to reduce future recurrence.
- Keep the answer practical, easy to understand, and aware of the farmer's location.
- If something is uncertain, say that clearly.

User message or transcript:
{user_text}
"""

    @staticmethod
    def text_prompt(location: str, language_name: str, user_input: str) -> str:
        return f"""
You are Farmora AI, a professional agricultural advisor.

Farmer location: {location}
Reply language: {language_name}

The user is asking a general farming question.
Answer in clean plain text only.
Do not use markdown.
Do not use asterisks.

If the question is general, use this structure:
Answer:
Practical Steps:
Best Practice:

If the question is specifically about a crop disease, deficiency, or plant issue without an image, you may also use:
Problem:
Treatment:
Prevention:

Rules:
- Answer the user's actual question directly.
- Make the answer practical and location-aware.
- Do not mention suppliers.
- Do not generate a PDF.
- Keep the wording simple and useful for farmers.

User question:
{user_input}
"""


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
    async def analyze_image(
        image_base64: str,
        location: str,
        user_text: str = "",
        language_name: str = "English",
    ) -> str:
        try:
            image_bytes = base64.b64decode(image_base64)

            response = await asyncio.to_thread(
                model.generate_content,
                [
                    PromptService.image_prompt(location, language_name, user_text),
                    {"mime_type": "image/jpeg", "data": image_bytes},
                ],
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
        language = (req.language or "en").strip().lower()
        language_name = get_language_name(language)

        user_input = (req.message or "").strip()
        if not user_input and transcript:
            user_input = transcript

        if req.imageBase64:
            result = await AIService.analyze_image(
                req.imageBase64,
                location_str,
                user_text=user_input,
                language_name=language_name,
            )
        else:
            if not user_input:
                user_input = "Give general farming guidance."
            prompt = PromptService.text_prompt(location_str, language_name, user_input)
            result = await AIService.generate_text(prompt)

        logger.info(f"[{request_id}] Success")

        return ChatResponse(
            id=request_id,
            response=result,
            timestamp=datetime.utcnow().isoformat(),
            location=location_str,
            hasAudio=has_audio,
            transcript=transcript,
            language=language,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again.",
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