from flask import Flask, request, send_file, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI
from pydub import AudioSegment
import tempfile
import os
import traceback
import logging
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)
CORS(app, expose_headers=['X-Response-Text-B64'])  # Changed header name

SYSTEM_PROMPT = "תשיב בקצרה בעברית, בקול ברור. תן מענה מהיר לשאלה בלבד."

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files["audio"]
    temp_in = None
    temp_in_path = None
    wav_path = None
    
    try:
        # Create temp file with unique name
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        temp_in_path = temp_in.name
        temp_in.close()  # Close the file handle
        
        audio_file.save(temp_in_path)
        logger.info(f"Audio saved to temporary file: {temp_in_path}")
        
        # Convert audio to WAV for Whisper
        audio = AudioSegment.from_file(temp_in_path)
        wav_path = temp_in_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")
        logger.info(f"Converted audio to WAV: {wav_path}")
        
        with open(wav_path, "rb") as f:
            # Use OpenAI's Whisper for transcription
            logger.info("Sending to Whisper API...")
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language="he"  # Hebrew
            )
            
            result = transcription.strip()
            logger.info(f"Transcription result: {result}")
            return jsonify({"transcription": result})
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Transcription error: {str(e)}\n{error_trace}")
        return jsonify({"error": f"Failed to transcribe: {str(e)}"}), 500
    
    finally:
        # Clean up temporary files
        try:
            if temp_in_path and os.path.exists(temp_in_path):
                os.unlink(temp_in_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

@app.route("/text", methods=["POST"])
def text():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        logger.info(f"Processing text request: {prompt}")
        chat_completion = client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:yahli-gal-personal:tut-bot-v2:BKqOkT5f",
            messages=[{"role": "user", "content": prompt}]
        )
        
        reply = chat_completion.choices[0].message.content
        logger.info(f"Response received: {reply[:50]}...")
        return jsonify({"reply": reply})
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Text error: {str(e)}\n{error_trace}")
        return jsonify({"error": f"Text request failed: {str(e)}"}), 500

@app.route("/voice-response", methods=["POST"])
def voice_response():
    """Optimized endpoint that gets AI response and creates TTS"""
    data = request.get_json()
    user_text = data.get("text", "")
    
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    
    temp_file = None
    
    try:
        logger.info(f"Voice response request: {user_text}")
        
        # First get the text response from GPT-4
        logger.info("Getting response from GPT-4...")
        chat_completion = client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:yahli-gal-personal:tut-bot-v2:BKqOkT5f",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        reply_text = chat_completion.choices[0].message.content
        logger.info(f"GPT-4 reply: {reply_text}")
        
        # Create a temporary file for the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = temp_file.name
        temp_file.close()  # Close the file handle
        
        # Generate TTS from the text response
        logger.info("Generating TTS...")
        speech = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=reply_text
        )
        
        speech.stream_to_file(temp_path)
        logger.info(f"TTS saved to: {temp_path}")
        
        # Base64 encode the Hebrew text to avoid Unicode issues in headers
        b64_text = base64.b64encode(reply_text.encode('utf-8')).decode('ascii')
        
        # Create response with the audio file and encoded text header
        response = make_response(send_file(temp_path, mimetype="audio/mpeg"))
        response.headers['X-Response-Text-B64'] = b64_text  # Use a new header name
        logger.info("Sending response to client")
        
        # Not removing temp file here - let OS handle it later
        # This prevents file busy errors on Windows
        
        return response
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Voice response error: {str(e)}\n{error_trace}")
        
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass
                
        return jsonify({"error": f"Voice response failed: {str(e)}"}), 500

@app.route("/speak", methods=["POST"])
def speak():
    """Legacy endpoint for TTS (kept for backward compatibility)"""
    data = request.get_json()
    user_text = data.get("text", "")
    
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Get response from GPT-4
        logger.info(f"Speak request: {user_text}")
        chat_completion = client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:yahli-gal-personal:tut-bot-v2:BKqOkT5f",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )
        
        reply_text = chat_completion.choices[0].message.content
        logger.info(f"GPT-4 reply: {reply_text}")
        
        # Generate speech from the text response
        speech = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=reply_text
        )
        
        # Create and save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = temp_file.name
        temp_file.close()
        
        speech.stream_to_file(temp_path)
        logger.info(f"TTS saved to: {temp_path}")
        
        return send_file(temp_path, mimetype="audio/mpeg")
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Speak error: {str(e)}\n{error_trace}")
        return jsonify({"error": f"Failed to generate speech: {str(e)}"}), 500

# Simple monitoring endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

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
