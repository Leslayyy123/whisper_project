"""
Simple FastAPI server for receiving audio alerts and saving to Supabase.

Run:
  uvicorn audio_server:app --host 127.0.0.1 --port 8000 --reload

Required environment variables:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY (preferred) or SUPABASE_ANON_KEY
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import Client, create_client


class AudioAlert(BaseModel):
    type: str
    quadrant: str
    confidence: float
    timestamp: str


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/SUPABASE_ANON_KEY "
                "environment variables."
            ),
        )
    return create_client(url, key)


app = FastAPI(title="Audio Alert Server")


@app.get("/status")
def status() -> dict[str, str]:
    return {"status": "running"}


@app.post("/audio-alert")
def audio_alert(payload: AudioAlert) -> dict[str, str]:
    supabase = get_supabase_client()
    data = {
        "type": payload.type,
        "quadrant": payload.quadrant,
        "confidence": payload.confidence,
        "timestamp": payload.timestamp,
    }
    result = supabase.table("audio_alerts").insert(data).execute()

    if getattr(result, "data", None) is None:
        raise HTTPException(status_code=500, detail="Failed to insert alert into audio_alerts.")

    return {"message": "Alert saved"}
