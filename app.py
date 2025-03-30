from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import uuid

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Your Flask server is running!"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    original_path = f"temp_{uuid.uuid4().hex}.webm"
    converted_path = f"converted_{uuid.uuid4().hex}.mp3"

    try:
        # Save uploaded WebM
        audio_file.save(original_path)

        # Convert to mp3 using pydub
        sound = AudioSegment.from_file(original_path)
        sound.export(converted_path, format="mp3")

        # Transcribe with OpenAI
        with open(converted_path, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)

        return jsonify({"transcription": transcript["text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(original_path):
            os.remove(original_path)
        if os.path.exists(converted_path):
            os.remove(converted_path)

@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",  # Male voice
            input=text,
        )

        output_path = "tts_output.mp3"
        response.stream_to_file(output_path)

        return send_file(output_path, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
