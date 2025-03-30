from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # ✅ Enable CORS
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)  # ✅ Allow all origins (for dev use only)

@app.route("/")
def home():
    return "Your Flask server is running!"

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
        return jsonify({"error": str(e)}), 500

@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # ✅ Use GPT-4 to generate response
        chat_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "אתה עוזר קולי אינטליגנטי ודובר עברית"},
                {"role": "user", "content": user_text}
            ]
        )
        bot_reply = chat_response["choices"][0]["message"]["content"]
        print("🧠 GPT-4 reply:", bot_reply)

        # ✅ Convert bot reply to speech
        tts_response = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",  # ✅ Man voice (deep, clear)
            input=bot_reply,
        )

        audio_path = "tts_output.mp3"
        tts_response.stream_to_file(audio_path)

        # ✅ Return bot reply text + audio
        return send_file(audio_path, mimetype="audio/mpeg", as_attachment=False,
                         download_name="bot_reply.mp3", conditional=True,
                         headers={"Bot-Reply": bot_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
