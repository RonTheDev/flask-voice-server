from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route("/")
def home():
    return "Your Flask server is running!"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    return jsonify({"transcription": "This is where Whisper result will go."})

@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    text = data.get("text", "")
    return jsonify({"audio_url": "https://dummy-audio-link.com/tts.mp3"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
