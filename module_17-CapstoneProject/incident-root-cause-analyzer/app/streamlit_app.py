"""Streamlit UI for the Incident Root Cause Analyzer - drives the FastAPI app
(GET /health, POST /incidents/analyze, GET /incidents/{id}/trace) via an
in-process fastapi.testclient.TestClient, so no separate `uvicorn` process
is needed alongside this one.

Run:
    streamlit run streamlit_app.py
"""

import sys
import time

import streamlit as st
from fastapi.testclient import TestClient

sys.stdout.reconfigure(encoding="utf-8")

from app import app
from tools import KNOWN_SERVICES

st.set_page_config(page_title="Incident RCA Analyzer", page_icon="🔎", layout="centered")

CONFIDENCE_COLOR = {"high": "green", "medium": "orange", "low": "red"}

INCIDENTS = [
    {
        "alert": "Elevated 502 errors and payment failures on /checkout",
        "services": ["api-gateway", "payments-service"],
        "start_time": "2026-08-30T09:10:00Z",
        "end_time": "2026-08-30T09:30:00Z",
    },
    {
        "alert": "Spike in 500/504 errors on /checkout",
        "services": ["api-gateway"],
        "start_time": "2026-09-01T03:45:00Z",
        "end_time": "2026-09-01T04:10:00Z",
    },
]


@st.cache_resource
def get_client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


if "history" not in st.session_state:
    st.session_state.history = []  # list of {incident_id, alert, report, step_count, trace_url}

client = get_client()

st.title("🔎 Incident Root Cause Analyzer")
st.caption("LangGraph agent (Gemini) + Tool use + RAG over runbooks + Langfuse tracing, behind FastAPI.")

with st.sidebar:
    st.subheader("API health")
    health = client.get("/health")
    if health.status_code == 200:
        st.success("FastAPI app: healthy")
    else:
        st.error(f"FastAPI app unhealthy ({health.status_code})")

    if st.session_state.history:
        st.subheader("Past analyses (this session)")
        for entry in reversed(st.session_state.history):
            st.caption(f"`{entry['incident_id'][:8]}` — {entry['alert']}")

st.subheader("1. Choose an incident")

seeded_labels = [incident["alert"] for incident in INCIDENTS]
choice = st.radio("Source", seeded_labels + ["Custom incident"], index=0)

if choice == "Custom incident":
    alert = st.text_input("Alert", placeholder="e.g. Elevated 502 errors on /checkout")
    services = st.multiselect("Services the alert mentions", KNOWN_SERVICES)
    col1, col2 = st.columns(2)
    start_time = col1.text_input("Start time (ISO 8601 UTC)", placeholder="2026-08-30T09:10:00Z")
    end_time = col2.text_input("End time (ISO 8601 UTC)", placeholder="2026-08-30T09:30:00Z")
    st.caption(
        "Note: the seeded log/metric data only covers the two incidents above (2026-08-30 and "
        "2026-09-01 windows) - a custom time window outside those will come back with no evidence found, "
        "which is still a real, honest exercise of the fetch_logs/fetch_metrics tools."
    )
    incident = {"alert": alert, "services": services, "start_time": start_time, "end_time": end_time}
else:
    incident = next(i for i in INCIDENTS if i["alert"] == choice)
    st.json(incident)

st.subheader("2. Run the agent")
ready = bool(incident.get("alert") and incident.get("start_time") and incident.get("end_time"))
run = st.button("Analyze incident", type="primary", disabled=not ready)

if run:
    with st.spinner("Agent is investigating (several sequential Gemini calls - can take up to a minute)..."):
        response = client.post("/incidents/analyze", json=incident)

    if response.status_code != 200:
        st.error(f"Request failed ({response.status_code}): {response.text}")
    else:
        result = response.json()
        st.session_state.history.append(
            {
                "incident_id": result["incident_id"],
                "alert": incident["alert"],
                "report": result["report"],
                "step_count": result["step_count"],
                "trace_url": result["trace_url"],
            }
        )
        st.session_state.selected = len(st.session_state.history) - 1

st.subheader("3. Result")

if not st.session_state.history:
    st.info("Run an analysis above to see the RCA report here.")
else:
    labels = [f"{i + 1}. {e['alert']} (`{e['incident_id'][:8]}`)" for i, e in enumerate(st.session_state.history)]
    default_index = st.session_state.get("selected", len(labels) - 1)
    picked = st.selectbox("Viewing result for:", labels, index=default_index)
    entry = st.session_state.history[labels.index(picked)]
    report = entry["report"]

    confidence = report.get("confidence", "unknown")
    st.markdown(f"**Confidence:** :{CONFIDENCE_COLOR.get(confidence, 'gray')}[{confidence.upper()}]")

    st.markdown("**Summary**")
    st.write(report["summary"])

    st.markdown("**Timeline**")
    st.dataframe(report["timeline"], width="stretch", hide_index=True)

    st.markdown("**Root cause**")
    st.write(report["root_cause"])

    st.markdown("**Evidence**")
    for item in report["evidence"]:
        st.markdown(f"- {item}")

    st.markdown("**Recommended fix**")
    st.write(report["recommended_fix"])

    st.caption(f"{entry['step_count']} agent steps")
    if entry["trace_url"]:
        st.markdown(f"[View full trace in Langfuse]({entry['trace_url']})")

    with st.expander("View raw JSON report"):
        st.json(report)

    with st.expander("View step-by-step audit trace (GET /incidents/{id}/trace)"):
        # Langfuse's own trace-read API lags its ingestion by ~10-25s after
        # this incident's tracer.finish() flush - a 404 right after analysis
        # means "not indexed yet", not "missing", so retry with backoff
        # before giving up (same reasoning as agent.py's _invoke_with_retry).
        trace = None
        for attempt in range(6):
            response = client.get(f"/incidents/{entry['incident_id']}/trace")
            if response.status_code == 200:
                trace = response.json()
                break
            if attempt < 5:
                with st.spinner("Waiting for Langfuse to finish indexing this trace..."):
                    time.sleep(4)

        if trace is None:
            st.info(
                "Trace isn't queryable yet (Langfuse indexing can take up to "
                "~30s after analysis). Collapse and reopen this section to try again, "
                f"or view it directly: {entry['trace_url']}"
            )
        else:
            for i, step in enumerate(trace, start=1):
                st.markdown(f"**Step {i} [{step['type']}] {step['name']}**")
                col1, col2 = st.columns(2)
                col1.caption("Input")
                col1.code(step.get("input") or "-", language=None)
                col2.caption("Output")
                col2.code(step.get("output") or "-", language=None)
