from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import which
import os
import uuid

# Set ffmpeg path for pydub
AudioSegment.converter = which("ffmpeg")

app = Flask(__name__)
CORS(app)

openai_client = OpenAI()

@app.route("/")
def index():
    return "Voice Server is Running"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    temp_filename = f"temp_{uuid.uuid4()}.webm"
    output_filename = temp_filename.replace(".webm", ".mp3")

    # Save and convert
    audio_path = os.path.join("/tmp", temp_filename)
    output_path = os.path.join("/tmp", output_filename)
    audio_file.save(audio_path)
    audio = AudioSegment.from_file(audio_path)
    audio.export(output_path, format="mp3")

    # Transcribe
    with open(output_path, "rb") as f:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="he"
        )

    return jsonify({"text": transcription.text})


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    bot_response = completion.choices[0].message.content

    # TTS
    tts_response = openai_client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=bot_response
    )

    output_path = f"/tmp/{uuid.uuid4()}.mp3"
    tts_response.stream_to_file(output_path)

    return send_file(output_path, mimetype="audio/mpeg", download_name="response.mp3"), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
