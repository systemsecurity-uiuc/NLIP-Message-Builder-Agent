import json
import os
import re
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


def normalize_requested_content(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split())
    patterns = [
        r"^create an nlip message asking(?: the [a-z0-9 -]+ agent)? to (.+)$",
        r"^create an nlip message that asks(?: the [a-z0-9 -]+ agent)? to (.+)$",
        r"^make an nlip message asking(?: the [a-z0-9 -]+ agent)? to (.+)$",
        r"^build an nlip message asking(?: the [a-z0-9 -]+ agent)? to (.+)$",
        r"^ask(?: the [a-z0-9 -]+ agent)? to (.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            request = match.group(1).strip()
            request = request.rstrip(".")
            return request[:1].upper() + request[1:] + "."

    direct_question = re.search(r"(what|why|how|when|where|who|which)\b.+", cleaned, flags=re.IGNORECASE)
    if direct_question:
        question = direct_question.group(0).strip()
        what_is_match = re.match(r"what\s+(.+)\s+is\.?$", question, flags=re.IGNORECASE)
        if what_is_match:
            subject = what_is_match.group(1).strip()
            return f"What is {subject}?"
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"
        return question[:1].upper() + question[1:]

    return cleaned


def build_nlip_message(prompt: str) -> dict[str, Any]:
    return {
        "messageType": "request",
        "format": "text",
        "subformat": "english",
        "content": normalize_requested_content(prompt),
    }


def ask_llm_to_build(prompt: str) -> dict[str, Any] | None:
    if not JETSTREAM_CHAT_URL:
        return None

    instruction = (
        "Convert the user request into exactly one ECMA-430-shaped NLIP JSON message. "
        "Allowed format values are text, token, structured, binary, location, and generic. "
        "For ordinary natural-language requests, use format text and subformat english. "
        "Do not include sender or receiver fields. Use keys messageType, format, "
        "subformat, content, and optional submessages. Return JSON only. "
        "Example: if the user says 'Create an NLIP message asking what NLIP is', "
        "return {\"messageType\":\"request\",\"format\":\"text\",\"subformat\":\"english\","
        "\"content\":\"What is NLIP?\"}.\n\n"
        f"User request: {prompt}"
    )
    if "chat/completions" in JETSTREAM_CHAT_URL:
        body = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You generate valid NLIP JSON envelopes."},
                {"role": "user", "content": instruction},
            ],
            "temperature": 0.1,
            "max_tokens": 250,
        }
    else:
        body = {
            "model": MODEL_NAME,
            "prompt": instruction,
            "temperature": 0.1,
            "max_tokens": 250,
            "stop": ["\n\n", "```"],
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

    choice = payload["choices"][0]
    if "message" in choice:
        text = choice["message"]["content"].strip()
    else:
        text = choice.get("text", "").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    parsed = json.loads(text)
    for key in ("format", "subformat", "content"):
        if key not in parsed:
            raise ValueError(f"LLM output missing required NLIP field: {key}")
    parsed.pop("sender", None)
    parsed.pop("receiver", None)

    allowed_formats = {"text", "token", "structured", "binary", "location", "generic"}
    if parsed.get("format") not in allowed_formats:
        parsed["format"] = "text"
        parsed["subformat"] = "english"
        parsed["content"] = normalize_requested_content(prompt)

    structured_subformats = {"json", "uri", "xml", "html"}
    if parsed.get("format") == "structured" and parsed.get("subformat") not in structured_subformats:
        parsed["subformat"] = "json"

    if parsed.get("format") == "text":
        parsed["subformat"] = "english"
        if not isinstance(parsed.get("content"), str):
            parsed["content"] = normalize_requested_content(prompt)
        elif parsed["content"].strip().lower().startswith(("create an nlip", "make an nlip", "build an nlip")):
            parsed["content"] = normalize_requested_content(prompt)

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
