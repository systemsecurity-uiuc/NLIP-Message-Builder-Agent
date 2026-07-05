# NLIP Message Builder Agent

Helper agent that converts a natural-language instruction into an ECMA-430-shaped NLIP message.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8081
```

## Optional Jetstream LLM

Set `JETSTREAM_CHAT_URL` to an OpenAI-compatible Jetstream inference endpoint to make the builder model-backed. If it is not set, the service falls back to a deterministic template.

```bash
export JETSTREAM_CHAT_URL="https://llm.jetstream-cloud.org/llama-4-scout/v1/chat/completions"
export NLIP_MODEL_NAME="llama-4-scout"
```

## Example

```bash
curl -s -X POST http://localhost:8081/nlip \
  -H 'Content-Type: application/json' \
  -d '{"messageType":"request","format":"text","subformat":"english","content":"Ask the knowledge agent what NLIP is."}'
```

The generated message does not include `sender` or `receiver`; routing is handled by the client endpoint selection, not by the NLIP message body.
