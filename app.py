from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
import tempfile
from pydub import AudioSegment

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Flask server is running!"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    
    # Save temp webm file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_temp:
        webm_path = webm_temp.name
        audio_file.save(webm_path)

    try:
        # Convert webm to wav using pydub
        audio = AudioSegment.from_file(webm_path, format="webm")
        wav_path = webm_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")

        # Transcribe
        with open(wav_path, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        return jsonify({"transcription": transcript["text"]})
    except Exception as e:
        print("❌ Transcription Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text,
        )
        output_path = "tts_output.mp3"
        response.stream_to_file(output_path)
        return send_file(output_path, mimetype="audio/mpeg")
    except Exception as e:
        print("❌ TTS Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
