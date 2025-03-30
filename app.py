from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import which
import os
import tempfile
import uuid

app = Flask(__name__)
CORS(app)

# Whisper + TTS config
client = OpenAI()
AudioSegment.converter = which("ffmpeg")

@app.route('/')
def home():
    return "Voice server is up!"

@app.route('/speak', methods=['POST'])
def speak():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, f"{uuid.uuid4()}.webm")
    output_path = input_path.replace(".webm", ".mp3")

    audio_file.save(input_path)
    sound = AudioSegment.from_file(input_path)
    sound.export(output_path, format="mp3")

    with open(output_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    text = transcript.text.strip()

    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text if text else "לא הבנתי, נסה שוב."
    )

    tts_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp3")
    response.stream_to_file(tts_path)

    return send_file(tts_path, mimetype='audio/mpeg', as_attachment=False, download_name="response.mp3", headers={"X-Transcript": text})
