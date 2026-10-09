import os
import shutil
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Original-Text", "X-Translated-Text"],
)

# Konfigurer Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

@app.post("/translate-audio")
async def translate_audio(
    file: UploadFile = File(...),
    source_lang: str = Form("no"),
    target_lang: str = Form("th"),
):
    # Behold filendelse slik at Gemini forstår lydformatet (f.eks. .m4a / .wav)
    file_ext = os.path.splitext(file.filename)[1] or ".m4a"
    temp_filename = f"temp_upload{file_ext}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Last opp lydfilen direkte til Gemini File API
        audio_file = genai.upload_file(path=temp_filename)

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

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )

        response = model.generate_content([audio_file, prompt])

        # Slett filen fra Gemini etter behandling
        try:
            genai.delete_file(audio_file.name)
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