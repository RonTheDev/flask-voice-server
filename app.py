from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import openai
import json
from pydub import AudioSegment
import tempfile
import os
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor
from functions import query_knowledgebase, tool_definitions
from system_prompt import get_system_prompt, ANSWER_PROMPT
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI and Flask
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)
CORS(app, expose_headers=['X-Response-Text-B64'])

executor = ThreadPoolExecutor(max_workers=4)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files["audio"]
    temp_in_path = None
    wav_path = None
    try:
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        temp_in_path = temp_in.name
        temp_in.close()
        audio_file.save(temp_in_path)

        audio = AudioSegment.from_file(temp_in_path)
        wav_path = temp_in_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")

        with open(wav_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language="he"
            )
            return jsonify({"transcription": transcription.strip()})
    except Exception as e:
        logger.error(f"Transcription error: {traceback.format_exc()}")
        return jsonify({"error": f"Failed to transcribe: {str(e)}"}), 500
    finally:
        if temp_in_path and os.path.exists(temp_in_path):
            os.unlink(temp_in_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

@app.route("/text", methods=["POST"])
def text():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        logger.info(f"Processing prompt: {prompt}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": get_system_prompt(tool_definitions)},
                {"role": "user", "content": prompt}
            ],
            tools=tool_definitions,
            tool_choice="auto"
        )

        tool_call = response.choices[0].message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = eval(tool_call.function.arguments)

        if tool_name == "query_knowledgebase":
            tool_result = query_knowledgebase(**tool_args)

            follow_up = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": ANSWER_PROMPT},
                    {"role": "user", "content": prompt},
                    response.choices[0].message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                         "content": json.dumps(tool_result, ensure_ascii=False)
                    }
                ]
            )
            return jsonify({"reply": follow_up.choices[0].message.content})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/text-stream", methods=["POST"])
def text_stream():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return Response("error: No prompt provided", mimetype="text/plain")

    def generate():
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": get_system_prompt(tool_definitions)},
                    {"role": "user", "content": prompt}
                ],
                tools=tool_definitions,
                tool_choice="auto"
            )

            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = eval(tool_call.function.arguments)

            if tool_name == "query_knowledgebase":
                tool_result = query_knowledgebase(**tool_args)

                stream = client.chat.completions.create(
                    model="gpt-4o",
                    stream=True,
                    messages=[
                        {"role": "system", "content": ANSWER_PROMPT},
                        {"role": "user", "content": prompt},
                        response.choices[0].message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                          "content": json.dumps(tool_result, ensure_ascii=False)
                        }
                    ]
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n[שגיאה: {str(e)}]"

    return Response(generate(), mimetype='text/plain')

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
