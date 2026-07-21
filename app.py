import json
import os
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


AGENT_NAME = "nlip-builder-agent"
MODEL_NAME = os.getenv("NLIP_MODEL_NAME", "llama-4-scout")
JETSTREAM_CHAT_URL = os.getenv("JETSTREAM_CHAT_URL", "")
JETSTREAM_API_KEY = os.getenv("JETSTREAM_API_KEY", "")


class NLIPMessage(BaseModel):
    messageType: str | None = None
    format: str
    subformat: str
    content: Any
    submessages: list[dict[str, Any]] = Field(default_factory=list)


app = FastAPI(title="NLIP Message Builder Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def nlip_response(content: Any) -> dict[str, Any]:
    return {
        "messageType": "response",
        "format": "structured",
        "subformat": "json",
        "content": content,
    }


def extract_prompt(message: NLIPMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, dict):
        for key in ("prompt", "query", "text"):
            value = message.content.get(key)
            if isinstance(value, str):
                return value
    raise HTTPException(
        status_code=400,
        detail="NLIP content must be text or a JSON object with prompt/query/text.",
    )


def build_nlip_message(prompt: str) -> dict[str, Any]:
    cleaned = prompt.strip()
    return {
        "messageType": "request",
        "format": "text",
        "subformat": "english",
        "content": cleaned,
    }


def ask_llm_to_build(prompt: str) -> dict[str, Any] | None:
    if not JETSTREAM_CHAT_URL:
        return None

    instruction = (
        "Convert the user request into exactly one ECMA-430-shaped NLIP JSON message. "
        "Do not include sender or receiver fields. Use keys messageType, format, "
        "subformat, content, and optional submessages. Return JSON only.\n\n"
        f"User request: {prompt}"
    )
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You generate valid NLIP JSON envelopes."},
            {"role": "user", "content": instruction},
        ],
        "temperature": 0.1,
    }
    headers = {"Content-Type": "application/json"}
    if JETSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {JETSTREAM_API_KEY}"

    req = urllib.request.Request(
        JETSTREAM_CHAT_URL,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    text = payload["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    parsed = json.loads(text)
    for key in ("format", "subformat", "content"):
        if key not in parsed:
            raise ValueError(f"LLM output missing required NLIP field: {key}")
    parsed.pop("sender", None)
    parsed.pop("receiver", None)
    return parsed


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "agent": AGENT_NAME,
        "model": MODEL_NAME,
        "llm_configured": bool(JETSTREAM_CHAT_URL),
    }


@app.post("/nlip")
def handle_nlip(message: NLIPMessage) -> dict[str, Any]:
    prompt = extract_prompt(message)
    try:
        generated = ask_llm_to_build(prompt)
    except Exception as error:
        generated = None
        llm_error = str(error)
    else:
        llm_error = None

    if generated is None:
        generated = build_nlip_message(prompt)

    content = {
        "agent": AGENT_NAME,
        "purpose": "Convert natural language into an ECMA-430-shaped NLIP message.",
        "original_prompt": prompt,
        "model": MODEL_NAME,
        "generated_nlip_message": generated,
    }
    if llm_error:
        content["llm_error"] = llm_error

    return nlip_response(
        content
    )
