from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ Flask server is running!"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    audio_path = "temp_audio.webm"
    audio_file.save(audio_path)

    try:
        with open(audio_path, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        return jsonify({"transcription": transcript["text"]})
    except Exception as e:
        print("❌ Transcription error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/speak", methods=["POST"])
def speak():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        print("🔊 Generating speech for:", text)

        response = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",  # Male voice
            input=text,
        )

        output_path = "tts_output.mp3"
        response.stream_to_file(output_path)

        print("✅ Audio generated successfully.")
        return send_file(output_path, mimetype="audio/mpeg")

    except Exception as e:
        print("❌ TTS generation error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
