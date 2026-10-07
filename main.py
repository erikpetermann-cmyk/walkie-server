import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Original-Text", "X-Translated-Text"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.post("/translate-audio")
async def translate_audio(
    file: UploadFile = File(...),
    source_lang: str = Form("no"),
    target_lang: str = Form("th"),
):
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Transkribering
        with open(temp_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=source_lang,
            )
        original_text = transcript.text.strip()
        print(f"\n[HØRT TALE ({source_lang})]: {original_text}")

        # 2. Oversettelse
        if target_lang == "th":
            system_prompt = (
                "Du er en tolk mellom norsk og thai. Oversett den norske setningen til naturlig, "
                "muntlig hverdagsthai slik folk faktisk snakker sammen. "
                "Bruk mannlig høflighetspartikkel (ครับ / khrap). Ikke oversett ordrett eller stivt. "
                "Svar KUN med oversettelsen i thai skrift."
            )
        else:
            system_prompt = (
                "Du er en tolk fra thai til norsk. Brukeren snakker muntlig thai / Isan-påvirket dagligtale. "
                "Gjør ditt beste for å forstå meningen selv om transkripsjonen har feil pga dialekt eller tonefall. "
                "Oversett meningen til god, naturlig norsk. Svar KUN med den norske oversettelsen."
            )

        # 3. Oversettelse med gpt-4o-mini
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_text},
            ],
            temperature=0.2,
        )
        translated_text = completion.choices[0].message.content.strip()
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