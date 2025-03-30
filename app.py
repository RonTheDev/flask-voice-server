from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from shutil import which
import openai
import os
import tempfile

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Fix for Render's pydub needing explicit ffmpeg path
AudioSegment.converter = which("ffmpeg")

@app.route("/")
def index():
    return "Voice server is live."

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        audio_file.save(temp_audio.name)
        temp_wav = temp_audio.name + ".mp3"

        sound = AudioSegment.from_file(temp_audio.name)
        sound.export(temp_wav, format="mp3")

    with open(temp_wav, "rb") as file:
        transcript = openai.audio.transcriptions.create(model="whisper-1", file=file)
        text = transcript.text

    response = openai.audio.speech.create(model="tts-1", voice="onyx", input=text)
    audio_data = response.read()

    return jsonify({"text": text, "audio": {"data": list(audio_data)}})

@app.route("/speak", methods=["POST"])
def speak():
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    response = openai.audio.speech.create(model="tts-1", voice="onyx", input=text)
    audio_data = response.read()
    return jsonify({"text": text, "audio": {"data": list(audio_data)}})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
