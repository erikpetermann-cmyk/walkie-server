import os
import json
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Original-Text", "X-Translated-Text"],
)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

class TranslationResult(BaseModel):
    original_text: str
    translated_text: str

@app.post("/translate-audio")
async def translate_audio(
    file: UploadFile = File(...),
    source_lang: str = Form("no"),
    target_lang: str = Form("th"),
    sender: str = Form("Erik"),
    recipient: str = Form("Alle"),
    channel: str = Form("CH-01"),
):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY mangler")

    try:
        audio_bytes = await file.read()
        mime_type = file.content_type

        # Sørg for gyldig MIME-type dersom telefonen sender tom type
        if not mime_type or mime_type == "application/octet-stream":
            filename = (file.filename or "").lower()
            if filename.endswith(".m4a") or filename.endswith(".mp4"):
                mime_type = "audio/mp4"
            elif filename.endswith(".wav"):
                mime_type = "audio/wav"
            elif filename.endswith(".3gp") or filename.endswith(".3gpp"):
                mime_type = "audio/3gpp"
            elif filename.endswith(".aac"):
                mime_type = "audio/aac"
            else:
                mime_type = "audio/mp4"

        if target_lang == "th":
            prompt = (
                "Transkriber norsk tale og oversett direkte til naturlig muntlig thai (bruk høflig partikkel ครับ/khrap)."
            )
        else:
            prompt = (
                "Transkriber thai/Isan tale og oversett direkte til naturlig muntlig norsk."
            )

        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type=mime_type,
        )

        # Bruker response_schema og 1000 tokens for å garantere fullført JSON
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TranslationResult,
            max_output_tokens=1000,
            temperature=0.2,
        )

        response = None
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[audio_part, prompt],
                    config=config,
                )
                break
            except Exception as e:
                err_msg = str(e)
                if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < 1:
                    time.sleep(0.3)
                    continue
                raise e

        data = json.loads(response.text)
        original_text = data.get("original_text", "").strip()
        translated_text = data.get("translated_text", "").strip()

        print(f"[{source_lang}->{target_lang}]: {original_text} -> {translated_text}")

        return {
            "original_text": original_text,
            "translated_text": translated_text,
            "sender": sender,
            "recipient": recipient,
            "channel": channel,
        }

    except Exception as e:
        print(f"Feil: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
