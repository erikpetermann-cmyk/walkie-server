import os
import json
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
        audio_bytes = await file.read()
        mime_type = file.content_type or "audio/m4a"

        # Korte, direkte instruksjoner for raskest mulig prosessering
        if target_lang == "th":
            prompt = (
                "Transkriber norsk tale og oversett direkte til naturlig muntlig thai (høflig form med ครับ/khrap). "
                'Svar kun gyldig JSON: {"original_text": "...", "translated_text": "..."}'
            )
        else:
            prompt = (
                "Transkriber thai/Isan tale og oversett direkte til naturlig muntlig norsk. "
                'Svar kun gyldig JSON: {"original_text": "...", "translated_text": "..."}'
            )

        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type=mime_type,
        )

        # Token-begrensning og lav temperatur for maksimal responshastighet
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=300,
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
