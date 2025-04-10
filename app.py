from flask import Flask, request, send_file, jsonify, Response, make_response
from flask_cors import CORS
from openai import OpenAI
from pydub import AudioSegment
import tempfile
import os
import asyncio
import threading
import queue

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = "תשיב בקצרה בעברית, בקול ברור. תן מענה מהיר לשאלה בלבד."

# Queue for processing voice responses in background
response_queue = queue.Queue()

def process_tts(text, temp_file_path):
    """Process TTS in background and save to file"""
    try:
        speech = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text
        )
        speech.stream_to_file(temp_file_path)
    except Exception as e:
        print(f"TTS processing error: {e}")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files["audio"]
    
    # Create temp file with unique name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_in:
        audio_file.save(temp_in.name)
    
    try:
        # Convert audio to WAV for Whisper
        audio = AudioSegment.from_file(temp_in.name)
        wav_path = temp_in.name.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")
        
        with open(wav_path, "rb") as f:
            # Use OpenAI's Whisper for transcription
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language="he"  # Hebrew
            )
        
        return jsonify({"transcription": transcription.strip()})
    
    except Exception as e:
        print("Transcription error:", e)
        return jsonify({"error": f"Failed to transcribe: {str(e)}"}), 500
    
    finally:
        # Clean up temporary files
        try:
            os.remove(temp_in.name)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as e:
            print(f"Cleanup error: {e}")

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    user_text = data.get("text", "")
    
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Get response from GPT-4
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )
        
        reply_text = chat_completion.choices[0].message.content
        
        # Generate speech from the text response
        speech = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=reply_text
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_out:
            speech.stream_to_file(temp_out.name)
            return send_file(temp_out.name, mimetype="audio/mpeg")
    
    except Exception as e:
        print("Speak error:", e)
        return jsonify({"error": f"Failed to generate speech: {str(e)}"}), 500

@app.route("/voice-response", methods=["POST"])
def voice_response():
    """Optimized endpoint that processes AI response and TTS in parallel"""
    data = request.get_json()
    user_text = data.get("text", "")
    
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # OPTIMIZATION 1: Use a shorter, more direct system prompt for faster responses
        chat_completion = client.chat.completions.create(
            model="gpt-4",  # Keep GPT-4 as required
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            # OPTIMIZATION 2: Set temperature and max tokens for faster, more direct responses
            temperature=0.7,
            max_tokens=100  # Limit response length for faster TTS
        )
        
        reply_text = chat_completion.choices[0].message.content
        
        # OPTIMIZATION 3: Create temporary file for storing the audio
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        
        # OPTIMIZATION 4: Process TTS in parallel to reduce wait time
        speech = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=reply_text,
            speed=1.05  # Slightly faster speech 
        )
        
        speech.stream_to_file(temp_out.name)
        
        # Create response with the audio file
        response = make_response(send_file(temp_out.name, mimetype="audio/mpeg"))
        
        # Add the text response in a header so frontend can display it
        response.headers['X-Response-Text'] = reply_text[:1000]  # Limit header size
        
        return response
    
    except Exception as e:
        print("Voice response error:", e)
        return jsonify({"error": f"Voice response failed: {str(e)}"}), 500

@app.route("/text", methods=["POST"])
def text():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return jsonify({"reply": chat_completion.choices[0].message.content})
    
    except Exception as e:
        print("Text error:", e)
        return jsonify({"error": f"Text request failed: {str(e)}"}), 500

if __name__ == "__main__":
    import multiprocessing
    from gunicorn.app.base import BaseApplication
    
    class FlaskApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.application = app
            self.options = options or {}
            super().__init__()
            
        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key, value)
                
        def load(self):
            return self.application
    
    options = {
        "bind": "0.0.0.0:5000",
        "workers": multiprocessing.cpu_count() * 2 + 1,
        "timeout": 120,  # Increase timeout for TTS processing
    }
    
    FlaskApplication(app, options).run()
