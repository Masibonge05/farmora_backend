# Farmora AI - Simple API 

import os
import base64
import asyncio
from datetime import datetime
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ================= CONFIG =================
PORT = int(os.getenv("PORT", 8000))
GEMINI_MODEL = "gemini-2.5-flash"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set")

genai.configure(api_key=GOOGLE_API_KEY)

vision_model = genai.GenerativeModel(GEMINI_MODEL)

# ================= APP =================
app = FastAPI(title="Farmora AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= MODELS =================
class LocationInfo(BaseModel):
    town: Optional[str] = ""
    province: Optional[str] = ""
    country: Optional[str] = "South Africa"

class RequestModel(BaseModel):
    imageBase64: str
    location: Optional[LocationInfo] = None


# ================= CORE ANALYSIS =================
async def analyze_image(image_base64: str, location: Optional[LocationInfo]):

    location_context = "South Africa"
    if location:
        location_context = f"{location.town}, {location.province}, {location.country}"

    prompt = f"""
You are an expert agricultural scientist in {location_context}.

Analyze the plant image and respond ONLY in JSON format like this:

{{
  "crop": "name of crop",
  "confidence": number (0-100),
  "problem": "what is wrong with the plant (or say healthy)",
  "solution": "how to fix the problem",
  "prevention": "how to prevent it in future"
}}

Keep answers clear, practical, and localized to {location_context}.
Do NOT include extra text outside JSON.
"""

    image_data = base64.b64decode(image_base64)
    image_part = {"mime_type": "image/jpeg", "data": image_data}

    response = await asyncio.to_thread(
        vision_model.generate_content,
        [prompt, image_part]
    )

    text = response.text.strip()

    # Try safe parsing
    try:
        import json
        data = json.loads(text)
    except:
        raise HTTPException(status_code=500, detail="AI response parsing failed")

    return data


# ================= ROUTE =================
@app.post("/analyze")
async def analyze(req: RequestModel):
    try:
        result = await analyze_image(req.imageBase64, req.location)

        return {
            "crop": result.get("crop", "Unknown"),
            "confidence": result.get("confidence", 0),
            "analysis": {
                "problem": result.get("problem", ""),
                "solution": result.get("solution", ""),
                "prevention": result.get("prevention", "")
            },
            "location": f"{req.location.town}, {req.location.province}, {req.location.country}"
            if req.location else "South Africa",
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================= HEALTH =================
@app.get("/")
def root():
    return {"status": "Farmora AI running 🚀"}