from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI
from pydub import AudioSegment
import tempfile
import os

app = Flask(__name__)
CORS(app)

client = OpenAI()

@app.route("/speak", methods=["POST"])
def speak():
    if "audio" not in request.files:
        return "No audio file", 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_in:
        audio_file.save(temp_in.name)

    try:
        audio = AudioSegment.from_file(temp_in.name)
        wav_path = temp_in.name.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")

        with open(wav_path, "rb") as f:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=f).text

        print("User said:", transcription)

        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": transcription}]
        )

        reply_text = chat_completion.choices[0].message.content

        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=reply_text
        )

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        response.stream_to_file(output_path)

        headers = {
            "x-transcript": transcription,
            "x-reply-text": reply_text
        }

        return send_file(output_path, mimetype="audio/mpeg", as_attachment=False, download_name="response.mp3", headers=headers)

    except Exception as e:
        print("Error:", e)
        return "Internal Server Error", 500
    finally:
        os.remove(temp_in.name)
        if os.path.exists(wav_path):
            os.remove(wav_path)

@app.route("/text", methods=["POST"])
def text():
    data = request.get_json()
    prompt = data.get("prompt", "")

    chat_completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return jsonify({"reply": chat_completion.choices[0].message.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
