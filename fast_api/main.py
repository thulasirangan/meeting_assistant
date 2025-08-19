import os
import uuid
import shutil
import subprocess
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv(override=True)
# === CONFIG ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WHISPER_MODEL = "whisper-large-v3"
LLAMA_MODEL = "openai/gpt-oss-120b"
TRANSCRIPTION_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# === INIT APP ===
app = FastAPI()

# CORS (adjust origin in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === UTILS ===
def convert_to_wav(uploaded_path: str) -> str:
    wav_path = f"/tmp/{uuid.uuid4()}.wav"
    command = [
        "ffmpeg", "-i", uploaded_path,
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        wav_path
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return wav_path
    except subprocess.CalledProcessError:
        raise RuntimeError("ffmpeg conversion failed")


def transcribe_with_groq(wav_path: str) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    with open(wav_path, "rb") as f:
        files = {
            "file": (os.path.basename(wav_path), f, "audio/wav"),
            "model": (None, WHISPER_MODEL)
        }
        res = requests.post(TRANSCRIPTION_ENDPOINT, headers=headers, files=files)
        if res.status_code == 200:
            return res.json()["text"]
        else:
            raise RuntimeError(f"Transcription failed: {res.text}")


def summarize_with_llama(transcript: str) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional meeting assistant. "
                    "Generate structured, clear Minutes of Meeting (MoM) from the transcript. "
                    "Include agenda items, key discussion points, decisions made, and action items."
                )
            },
            {"role": "user", "content": transcript}
        ],
        "temperature": 0.4
    }
    res = requests.post(CHAT_ENDPOINT, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"Summarization failed: {res.text}")


# === ROUTE ===
@app.post("/api/process")
async def process_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save uploaded file
    temp_input_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        wav_path = convert_to_wav(temp_input_path)
        transcript = transcribe_with_groq(wav_path)
        summary = summarize_with_llama(transcript)

        return JSONResponse({
            "transcript": transcript,
            "minutes_of_meeting": summary
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
