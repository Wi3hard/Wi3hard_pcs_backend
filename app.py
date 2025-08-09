import os
import time
import uuid
import base64
import tempfile
import json
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# === Configuration via environment variables ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")        # Optional fallback
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")        # Use this for Gemini
BACKEND_SECRET = os.getenv("BACKEND_SECRET", "devsecret")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-pro")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", (
    "You are Wi3hard PCS with AI, an expert tutor in Peace & Conflict Studies for Nigerian university "
    "students and lecturers. Always answer in clear, academic, exam-ready language. If a user asks for a "
    "\"summary\", produce concise bullet points. If asked for \"analysis\", include causes, actors, timeline, "
    "and consequences. For \"full\" responses, provide detailed explanations, case comparisons, and references when possible. "
    "When an image is provided, extract the text/diagram info and respond to it directly. Keep tone professional."
))

# === Utility helpers ===
def save_base64_image_to_tempfile(image_b64):
    header_sep = "base64,"
    if header_sep in image_b64:
        image_b64 = image_b64.split(header_sep)[1]
    binary = base64.b64decode(image_b64)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(binary)
    tmp.flush()
    tmp.close()
    return tmp.name

# === Gemini integration (using google-genai SDK) ===
def call_gemini_api(payload_text, image_bytes=None, mode="auto"):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set on server.")
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError("google-genai SDK not installed. pip install google-genai") from e
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = []
    if image_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.flush()
        tmp.close()
        try:
            uploaded_file = client.files.upload(file=tmp.name)
            contents.append(uploaded_file)
        finally:
            pass
    if payload_text and payload_text.strip():
        contents.append(payload_text)
    try:
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    except Exception:
        config = None
    model_name = DEFAULT_MODEL or "gemini-2.5-pro"
    if config:
        response = client.models.generate_content(model=model_name, contents=contents, config=config)
    else:
        response = client.models.generate_content(model=model_name, contents=contents)
    text = ""
    try:
        text = response.text
    except Exception:
        try:
            text = str(response)
        except Exception:
            text = "No textual response from Gemini."
    return {"reply_text": text, "raw": response, "reply_type": "analysis"}

# === OpenAI fallback ===
def call_openai_chat(system_prompt, user_text, model="gpt-4o", max_tokens=800, temperature=0.2):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set on server.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    text = ""
    try:
        text = j["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(j)
    return {"reply_text": text, "raw": j, "reply_type": "analysis"}

# === Main analyze endpoint ===
@app.route("/analyze", methods=["POST"])
def analyze():
    client_secret = request.headers.get("x-backend-secret")
    if BACKEND_SECRET != "devsecret" and client_secret != BACKEND_SECRET:
        return jsonify({"status":"error","message":"unauthorized"}), 401
    data = request.get_json(force=True)
    if not data:
        return jsonify({"status":"error","message":"invalid json payload"}), 400
    user_id = data.get("user_id", str(uuid.uuid4()))
    mode = data.get("mode", "auto")
    text = data.get("message_text", "")
    image_b64 = data.get("image_base64")
    if mode != "auto":
        user_text = f"[MODE: {mode.upper()}]\n\n{text}"
    else:
        user_text = text
    temp_image_path = None
    image_bytes = None
    if image_b64:
        try:
            temp_image_path = save_base64_image_to_tempfile(image_b64)
            with open(temp_image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            return jsonify({"status":"error","message":"bad image base64", "error": str(e)}), 400
    try:
        if GEMINI_API_KEY:
            ai_resp = call_gemini_api(payload_text=user_text, image_bytes=image_bytes, mode=mode)
            used = "gemini"
        elif OPENAI_API_KEY:
            ai_resp = call_openai_chat(SYSTEM_PROMPT, user_text)
            used = "openai"
        else:
            return jsonify({"status":"error","message":"No AI provider configured. Set GEMINI_API_KEY or OPENAI_API_KEY on the server."}), 500
    except Exception as e:
        return jsonify({"status":"error","message":"AI provider error", "error": str(e)}), 500
    finally:
        try:
            if temp_image_path:
                os.remove(temp_image_path)
        except Exception:
            pass
    return jsonify({
        "status":"ok",
        "reply_text": ai_resp.get("reply_text"),
        "reply_type": ai_resp.get("reply_type", "analysis"),
        "meta": {"used_provider": used}
    })

# === Health endpoint ===
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "time": int(time.time())})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
