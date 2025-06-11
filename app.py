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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
        tool_args = json.loads(tool_call.function.arguments)
        logger.debug(f"Tool call arguments: {tool_args}")

        if tool_name == "query_knowledgebase":
            tool_result = query_knowledgebase(**tool_args)
            logger.debug(f"Tool result: {tool_result}")

            follow_up_messages = [
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
            logger.debug(f"Follow up messages: {follow_up_messages}")

            follow_up = client.chat.completions.create(
                model="gpt-4o",
                messages=follow_up_messages
            )
            logger.debug(f"Model reply: {follow_up.choices[0].message.content}")
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
            tool_args = json.loads(tool_call.function.arguments)
            logger.debug(f"[stream] Tool call arguments: {tool_args}")

            if tool_name == "query_knowledgebase":
                tool_result = query_knowledgebase(**tool_args)
                logger.debug(f"[stream] Tool result: {tool_result}")

                follow_up_messages = [
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
                logger.debug(f"[stream] Follow up messages: {follow_up_messages}")


                stream = client.chat.completions.create(
                    model="gpt-4o",
                    stream=True,
                       messages=follow_up_messages
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content_to_yield = chunk.choices[0].delta.content
                        logger.debug(f"[stream] Yielding chunk: {content_to_yield}")
                        yield content_to_yield
        except Exception as e:
            logger.error(f"[stream] Error during generation: {traceback.format_exc()}")
            yield f"\n[שגיאה: {str(e)}]"

    return Response(generate(), mimetype='text/plain')

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/speak", methods=["POST"])
def speak():
    try:
        data = request.get_json()
        user_text = data.get("text", "")
        if not user_text:
            return Response("error: No text provided", mimetype="text/plain")

        logger.info(f"Generating GPT reply for: {user_text}")

        # STEP 1: Get GPT response (not echo)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": get_system_prompt(tool_definitions)},
                {"role": "user", "content": user_text}
            ],
            tools=tool_definitions,
            tool_choice="auto"
        )

        tool_call = response.choices[0].message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        tool_result = query_knowledgebase(**tool_args)

        followup_messages = [
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": user_text},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False)
            }
        ]

        followup = client.chat.completions.create(
            model="gpt-4o",
            messages=followup_messages
        )

        reply_text = followup.choices[0].message.content.strip()

        logger.info(f"TTS final reply: {reply_text}")

        # STEP 2: Generate speech from GPT reply
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=reply_text,
            response_format="opus"
        )

        # STEP 3: Stream audio + send base64-encoded reply text header
        def audio_stream():
            try:
                for chunk in tts_response.iter_bytes(chunk_size=4096):
                    yield chunk
            except Exception as e:
                logger.error(f"TTS streaming error: {traceback.format_exc()}")
                yield b''

        import base64
        b64_reply = base64.b64encode(reply_text.encode("utf-8")).decode("utf-8")
        resp = Response(audio_stream(), mimetype="audio/mpeg")
        resp.headers["X-Response-Text-B64"] = b64_reply
        return resp

    except Exception as e:
        logger.error(f"TTS endpoint error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
