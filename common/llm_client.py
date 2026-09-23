"""Shared client for talking to Google's Gemini API over REST.

Used by every Gemini-backed module, including the module_17 capstone.
The LangChain/LangGraph modules (8-10 and 17) get their chat model from
`langchain_chat_model()` below instead of calling `chat()` directly.

Chat/generation uses `gemini-3.1-flash-lite`; embeddings use
`gemini-embedding-001`. Reads GEMINI_API_KEY from the `.env` file next
to this module (via python-dotenv).

`chat()` returns a plain dict shaped like
    {"message": {"role": "assistant", "content": str, "tool_calls": [...]},
     "done_reason": "stop" | "length" | ...,
     "prompt_eval_count": int, "eval_count": int}
so callers read `response["message"]["content"]` for the reply text.
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")  # model output may include Unicode the console codepage can't render

load_dotenv(Path(__file__).parent / ".env")

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

MODEL = "gemini-3.1-flash-lite"
CHAT_URL = f"{BASE_URL}/{MODEL}:generateContent"
STREAM_URL = f"{BASE_URL}/{MODEL}:streamGenerateContent"

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_URL = f"{BASE_URL}/{EMBED_MODEL}:batchEmbedContents"
EMBED_BATCH_SIZE = 100  # batchEmbedContents accepts at most 100 requests per call

HEADERS = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

# Gemini occasionally returns a transient server error (e.g. 503 "high
# demand") or a free-tier rate limit (429) - neither reflects a bad
# request. A 503 clears up within seconds; a 429 is a per-minute quota, so
# its response carries a RetryInfo `retryDelay` saying how long to wait -
# honored when present, exponential backoff otherwise.
RETRYABLE_STATUS = {429, 500, 503}
MAX_RETRIES = 5
BASE_DELAY = 2.0

FINISH_REASONS = {"STOP": "stop", "MAX_TOKENS": "length"}


def _retry_delay(response: requests.Response, attempt: int) -> float:
    try:
        for detail in response.json()["error"]["details"]:
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(detail["retryDelay"].rstrip("s")) + random.uniform(1, 2)
    except (ValueError, KeyError, TypeError):
        pass
    return BASE_DELAY * (2**attempt) + random.uniform(0, 1)


def _post(url: str, payload: dict, stream: bool = False, params: dict | None = None) -> requests.Response:
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=HEADERS, json=payload, params=params, stream=stream, timeout=120)
        except requests.Timeout:  # an occasional stalled request - same treatment as a 503
            if attempt == MAX_RETRIES:
                raise
            print(f"  [retry] Gemini call timed out; retrying (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
            continue
        if response.status_code not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
            response.raise_for_status()
            return response
        delay = _retry_delay(response, attempt)
        response.close()
        print(f"  [retry] Gemini call failed ({response.status_code}); retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
        time.sleep(delay)


def _to_gemini_contents(messages: list[dict]) -> tuple[dict | None, list[dict]]:
    """Convert an OpenAI-style messages list into Gemini's
    (systemInstruction, contents) pair. System messages become the
    system instruction; `assistant` becomes Gemini's `model` role; `tool`
    results become functionResponse parts answering the preceding
    assistant turn's tool calls."""
    system_texts = []
    contents = []
    pending_call_names: list[str] = []

    for m in messages:
        role = m["role"]
        if role == "system":
            system_texts.append(m["content"])
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif role == "assistant":
            if m.get("_gemini_parts"):
                parts = m["_gemini_parts"]  # original parts, incl. thought signatures Gemini requires back
            else:
                parts = [{"text": m["content"]}] if m.get("content") else []
                for call in m.get("tool_calls") or []:
                    parts.append({"functionCall": {"name": call["function"]["name"], "args": call["function"]["arguments"]}})
            pending_call_names = [p["functionCall"]["name"] for p in parts if "functionCall" in p]
            contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            name = m.get("name") or m.get("tool_name") or (pending_call_names.pop(0) if pending_call_names else "tool")
            part = {"functionResponse": {"name": name, "response": {"result": m["content"]}}}
            previous = contents[-1] if contents else None
            if previous and previous["role"] == "user" and "functionResponse" in previous["parts"][0]:
                previous["parts"].append(part)  # all results for one model turn go back together
            else:
                contents.append({"role": "user", "parts": [part]})

    system_instruction = {"parts": [{"text": "\n\n".join(system_texts)}]} if system_texts else None
    return system_instruction, contents


def _build_payload(messages: list[dict], generation_config: dict, tools: list[dict] | None = None) -> dict:
    system_instruction, contents = _to_gemini_contents(messages)
    payload = {"contents": contents, "generationConfig": generation_config}
    if system_instruction:
        payload["systemInstruction"] = system_instruction
    if tools:
        payload["tools"] = [{"functionDeclarations": [t["function"] for t in tools]}]
    return payload


def chat(
    messages: list[dict],
    temperature: float = 0.7,
    seed: int | None = None,
    json_mode: bool = False,
    num_predict: int | None = None,
    tools: list[dict] | None = None,
) -> dict:
    """Call Gemini and return the response dict described in the module
    docstring - includes Gemini's own token counts (prompt_eval_count,
    eval_count). Pass json_mode=True to constrain output to valid JSON.
    num_predict caps the number of generated tokens (maxOutputTokens).
    tools is a list of OpenAI-style function-tool definitions; if the
    model decides to call one, the response's message["tool_calls"] is
    populated, with `arguments` already parsed as a dict."""
    generation_config = {"temperature": temperature}
    if seed is not None:
        generation_config["seed"] = seed
    if num_predict is not None:
        generation_config["maxOutputTokens"] = num_predict
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    data = _post(CHAT_URL, _build_payload(messages, generation_config, tools)).json()

    candidate = (data.get("candidates") or [{}])[0]
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    tool_calls = [
        {"function": {"name": p["functionCall"]["name"], "arguments": p["functionCall"].get("args", {})}}
        for p in parts
        if "functionCall" in p
    ]

    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["_gemini_parts"] = parts

    usage = data.get("usageMetadata", {})
    finish_reason = candidate.get("finishReason", "STOP")
    return {
        "message": message,
        "done_reason": FINISH_REASONS.get(finish_reason, finish_reason.lower()),
        "prompt_eval_count": usage.get("promptTokenCount", 0),
        "eval_count": usage.get("candidatesTokenCount", 0),
    }


def embed(texts: str | list[str], task_type: str | None = None) -> list[list[float]]:
    """Embed one or more strings via Gemini's batchEmbedContents endpoint.
    Always returns a list of vectors, one per input string (even for a
    single string input - index [0] to get that one vector). Truncated
    (768-dim) Gemini embeddings aren't unit length, so each vector is
    L2-normalized here - dot product and cosine similarity are then
    numerically identical downstream."""
    payload_input = [texts] if isinstance(texts, str) else texts

    vectors = []
    for start in range(0, len(payload_input), EMBED_BATCH_SIZE):
        batch = payload_input[start : start + EMBED_BATCH_SIZE]
        requests_body = []
        for text in batch:
            request = {"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": text}]}, "output_dimensionality": EMBED_DIM}
            if task_type:
                request["task_type"] = task_type
            requests_body.append(request)
        response = _post(EMBED_URL, {"requests": requests_body})
        vectors.extend(e["values"] for e in response.json()["embeddings"])

    normalized = []
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        normalized.append([x / norm for x in v])
    return normalized


def stream_chat(messages: list[dict]):
    """Call Gemini with streaming enabled; yields each content piece as
    it arrives (Server-Sent Events, one JSON chunk per `data:` line)."""
    payload = _build_payload(messages, {})
    with _post(STREAM_URL, payload, stream=True, params={"alt": "sse"}) as response:
        for raw_line in response.iter_lines():
            line = raw_line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[len("data:") :].strip())
            candidate = (chunk.get("candidates") or [{}])[0]
            for part in candidate.get("content", {}).get("parts", []):
                piece = part.get("text", "")
                if piece and not part.get("thought"):
                    yield piece


def langchain_chat_model(**kwargs):
    """LangChain's Gemini chat model (`ChatGoogleGenerativeAI`), configured
    with this client's MODEL and GEMINI_API_KEY - for the LangChain/LangGraph
    modules, which orchestrate through LangChain's own model interface
    rather than `chat()`. Extra kwargs (temperature, response_mime_type, ...)
    pass straight through. Imported lazily, so modules that never call this
    don't need `langchain-google-genai` installed."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=MODEL, api_key=GEMINI_API_KEY, **kwargs)
