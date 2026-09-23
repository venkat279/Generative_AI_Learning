"""Scripted walkthrough exercising every endpoint in app.py, using
FastAPI's TestClient (no real network server needed)

Run:
    python demo.py
"""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
HEADERS = {"X-API-Key": "demo-key"}


def show(label: str, response) -> None:
    print(f"\n{label}: {response.status_code}")
    content_type = response.headers.get("content-type", "")
    print(response.json() if content_type.startswith("application/json") else response.text[:300])


def main() -> None:
    show("GET /health", client.get("/health"))

    show("POST /chat (no api key -> 401)", client.post("/chat", json={"message": "hi"}))

    chat_1 = client.post("/chat", headers=HEADERS, json={"message": "My name is Alex.", "tenant_id": "acme"})
    show("POST /chat (turn 1)", chat_1)
    conversation_id = chat_1.json()["conversation_id"]

    chat_2 = client.post(
        "/chat",
        headers=HEADERS,
        json={"message": "What's my name?", "tenant_id": "acme", "conversation_id": conversation_id},
    )
    show("POST /chat (turn 2, same conversation_id -> session/state working)", chat_2)

    other_tenant = client.post(
        "/chat",
        headers=HEADERS,
        json={"message": "What's my name?", "tenant_id": "other-tenant", "conversation_id": conversation_id},
    )
    show("POST /chat (same conversation_id, DIFFERENT tenant -> should NOT know the name)", other_tenant)

    print("\nPOST /chat/stream (SSE):")
    with client.stream("POST", "/chat/stream", headers=HEADERS, json={"message": "Count from 1 to 5."}) as stream_response:
        for line in stream_response.iter_lines():
            if line:
                print(f"  {line}")

    with client.websocket_connect("/ws/chat?tenant_id=acme") as ws:
        greeting = ws.receive_json()
        print(f"\nWS /ws/chat connected: {greeting}")
        ws.send_text("Say hi in one word.")
        print(f"WS reply: {ws.receive_text()}")

    job = client.post("/jobs/report", headers=HEADERS, params={"topic": "weekly deployment activity"})
    show("POST /jobs/report", job)
    job_id = job.json()["job_id"]
    show("GET /jobs/{job_id} (report)", client.get(f"/jobs/{job_id}"))

    upload = client.post(
        "/files/upload",
        headers=HEADERS,
        files={
            "file": (
                "notes.txt",
                b"The payments service migrated to Kafka last quarter and reduced retry storms significantly.",
                "text/plain",
            )
        },
    )
    show("POST /files/upload", upload)
    upload_job_id = upload.json()["job_id"]
    show("GET /jobs/{job_id} (file summary)", client.get(f"/jobs/{upload_job_id}"))

    bad_upload = client.post(
        "/files/upload",
        headers=HEADERS,
        files={"file": ("image.png", b"\x89PNG...", "image/png")},
    )
    show("POST /files/upload (unsupported type -> 415)", bad_upload)

    # The rate limit runs before the handler's check, so a blocked call reads 429 instead of 415.
    print("\nRate limiting (6 quick calls against a 5-per-10s limit, via repeated bad uploads):")
    for i in range(6):
        r = client.post(
            "/files/upload",
            headers=HEADERS,
            files={"file": ("image.png", b"\x89PNG...", "image/png")},
        )
        print(f"  call {i + 1}: {r}")


if __name__ == "__main__":
    main()
