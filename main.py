import os
import json
import base64
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

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
        # Les lyden rett inn i minnet som bytes
        audio_bytes = await file.read()
        mime_type = file.content_type or "audio/m4a"

        if target_lang == "th":
            prompt = (
                "Hør nøye på lydfilen på norsk. "
                "1. Transkriber den norske teksten nøyaktig. "
                "2. Oversett setningen til naturlig, muntlig hverdagsthai slik folk snakker. "
                "Bruk mannlig høflighetspartikkel (ครับ / khrap). "
                "Returner svaret KUN som et gyldig JSON-objekt i dette formatet: "
                '{"original_text": "norsk tekst her", "translated_text": "thai tekst her"}'
            )
        else:
            prompt = (
                "Hør nøye på lydfilen på thai / Isan / lokal dialekt fra Sakon Nakhon (phasa Yo). "
                "1. Transkriber hva som blir sagt. "
                "2. Forstå meningen og hensikten bak dialekten, og oversett til naturlig, flytende og uformell norsk tale. "
                "Returner svaret KUN som et gyldig JSON-objekt i dette formatet: "
                '{"original_text": "thai tekst her", "translated_text": "norsk tekst her"}'
            )

        # Send lyden direkte i minnet (uten filopplasting)
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type=mime_type,
        )

        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=[audio_part, prompt],
                    config={"response_mime_type": "application/json"}
                )
                break
            except Exception as e:
                err_msg = str(e)
                if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < 2:
                    time.sleep(1.0)
                    continue
                raise e

        data = json.loads(response.text)
        original_text = data.get("original_text", "").strip()
        translated_text = data.get("translated_text", "").strip()

        print(f"\n[HØRT ({source_lang})]: {original_text}")
        print(f"[OVERSETTELSE ({target_lang})]: {translated_text}")

        return {
            "original_text": original_text,
            "translated_text": translated_text,
            "sender": sender,
            "recipient": recipient,
            "channel": channel
        }

    except Exception as e:
        print(f"Feil: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
