import os
import shutil
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Original-Text", "X-Translated-Text"],
)

# Initialiser Google GenAI klient med API-nøkkelen (støtter nye AQ-nøkler)
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
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY er ikke konfigurert")

    file_ext = os.path.splitext(file.filename)[1] or ".m4a"
    temp_filename = f"temp_upload{file_ext}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Last opp lydfilen til Gemini
        audio_file = client.files.upload(file=temp_filename)

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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[audio_file, prompt],
            config={
                "response_mime_type": "application/json"
            }
        )

        # Slett midlertidig lydfil fra Google etter bruk
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass

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
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
