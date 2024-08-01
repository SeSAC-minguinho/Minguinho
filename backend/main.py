from fastapi import FastAPI, Form, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import io
from datetime import datetime
from gtts import gTTS
from pydub import AudioSegment
import numpy as np
import base64
import uvicorn
import requests
import os

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

origins = [
    "http://localhost:8080",
    "http://localhost:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()


def STT(audio):
    buffer = io.BytesIO(audio)
    buffer.name = "input.wav"
    transcript = client.audio.transcriptions.create(
        file=buffer,
        model="whisper-1",
        language="ko",
        response_format="text",
        # timestamp_granularities=["word"]
    )
    return transcript


def ask_gpt(prompt, model="gpt-4o-mini"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "유구한 세월을 살아온 흑염룡으로서 허세와 난해한 단어들을 사용해 30단어 내외로 답변한다."},
            {"role": "user", "content": prompt}
        ]
    )
    print("Assistant: " + completion.choices[0].message.content)
    return completion.choices[0].message.content


def TTS(response):
    tts = gTTS(text=response, lang="ko")
    mp3_io = io.BytesIO()
    tts.write_to_fp(mp3_io)
    mp3_io.seek(0)
    return mp3_io.read()


def openai_tts(response_str, voice='nova'):
    if voice not in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
        raise ValueError("voice must be one of 'alloy echo fable onyx nova shimmer'")
    response_voice = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=response_str,
    )
    return response_voice.read()


def gcloud_tts(response_str, voice='ko-KR-Wavenet-A', api_key=os.environ["GOOGLE_TTS_API_KEY"]):
    url = "https://texttospeech.googleapis.com/v1/text:synthesize?key=" + api_key
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "input": {"text": response_str},
        "voice": {"languageCode": "ko-KR", "name": voice},
        "audioConfig": {"audioEncoding": "MP3"}
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    audio_content = response.json()["audioContent"]
    return base64.b64decode(audio_content)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    api_hostname = os.getenv("API_HOSTNAME", "localhost:8000")
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read().replace("{{ api_hostname }}", api_hostname)
        return HTMLResponse(content=html_content)


@app.get("/config")
async def return_config():
    ret_json = {"http_protocol": os.environ["HTTP_PROTOCOL"],
                "api_hostname": os.environ["API_HOSTNAME"],
                }
    return ret_json


@app.post("/process_audio_chat")
async def process_audio_chat(audio: UploadFile = File(...)):
    contents = await audio.read()
    float32_array = np.frombuffer(contents, dtype=np.float32)
    int16_array = (float32_array * 32767).astype(np.int16)
    byte_data = int16_array.tobytes()
    audio_segment = AudioSegment(
        data=byte_data,
        sample_width=2,  # 2 bytes for int16
        frame_rate=16000,  # Assuming a sample rate of 16000 Hz
        channels=1  # Assuming mono audio
    )
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    print("Exported to WAV format")

    question = STT(wav_io.read())
    print("STT result:", question)

    now = datetime.now().strftime("%H:%M")
    chat_history.append(("user", now, question))
    messages.append({"role": "user", "content": question})

    response = ask_gpt(question)
    messages.append({"role": "system", "content": response})

    now = datetime.now().strftime("%H:%M")
    chat_history.append(("bot", now, response))

    tts = os.environ["TTS_SERVICE"]
    voice_name = os.environ["TTS_VOICE"]
    print(f"tts: {tts}, voice_name: {voice_name}")

    if tts == "gtts":
        tts_data = TTS(response)
    elif tts == "google":
        tts_data = gcloud_tts(response, voice_name)
    else:
        tts_data = openai_tts(response, voice_name)

    # Encode the MP3 data in base64
    mp3_base64 = base64.b64encode(tts_data).decode('utf-8')

    return {"success": True, "question": question, "response": response, "audio_file": mp3_base64}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
