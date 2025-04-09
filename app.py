from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI
from pydub import AudioSegment
import tempfile
import os

app = Flask(__name__)
CORS(app)

client = OpenAI()

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_in:
        audio_file.save(temp_in.name)

    try:
        audio = AudioSegment.from_file(temp_in.name)
        wav_path = temp_in.name.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")

        with open(wav_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="he"  # ✅ Force Hebrew transcription
            ).text

        return jsonify({"transcription": transcription})

    except Exception as e:
        print("Transcription error:", e)
        return jsonify({"error": "Failed to transcribe"}), 500
    finally:
        os.remove(temp_in.name)
        if os.path.exists(wav_path):
            os.remove(wav_path)

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    user_text = data.get("text", "")

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "ענה בעברית בלבד."},  # ✅ Always respond in Hebrew
                {"role": "user", "content": user_text}
            ]
        )

        reply_text = chat_completion.choices[0].message.content

        speech = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=reply_text
        )

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        speech.stream_to_file(output_path)

        return send_file(output_path, mimetype="audio/mpeg")

    except Exception as e:
        print("Speak error:", e)
        return jsonify({"error": "Failed to generate speech"}), 500

@app.route("/text", methods=["POST"])
def text():
    data = request.get_json()
    prompt = data.get("prompt", "")

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "ענה בעברית בלבד."},  # ✅ Force Hebrew for text mode too
                {"role": "user", "content": prompt}
            ]
        )
        return jsonify({"reply": chat_completion.choices[0].message.content})

    except Exception as e:
        print("Text error:", e)
        return jsonify({"error": "Text request failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
