"""LangGraph agent for incident investigation.

An `agent` node (Gemini, bound to fetch_logs/fetch_metrics/search_runbooks)
decides which tools to call and when it has enough evidence, looping
through a `tools` node until it stops requesting calls. A separate,
tool-free `extract` node then turns its plain-language summary into a
structured JSON RCA report.

Every model turn is logged to Langfuse as a "generation" observation and
every tool call as a "tool" observation (tracer.py) - reconstructed by
`_log_transcript` from the finished message list once the run completes.
"""

import json
import random
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from llm_client import MODEL, langchain_chat_model  # noqa: E402

from runbook_rag import search_runbooks  # noqa: E402
from tools import KNOWN_SERVICES, fetch_logs, fetch_metrics  # noqa: E402
from tracer import IncidentTracer  # noqa: E402

MODEL_NAME = MODEL
TOOLS = [fetch_logs, fetch_metrics, search_runbooks]

# Gemini occasionally returns a transient server error (GoogleAPIError,
# e.g. 503 "high demand") or a free-tier rate limit (GoogleRateLimitError,
# 429) - neither reflects a bad request, and both clear up within seconds,
# so a short retry-with-backoff is enough.
_RETRYABLE_EXCEPTIONS = (GoogleAPIError, GoogleRateLimitError)


def _invoke_with_retry(invoke, max_retries: int = 3, base_delay: float = 2.0):
    for attempt in range(max_retries + 1):
        try:
            return invoke()
        except _RETRYABLE_EXCEPTIONS as exc:
            if attempt == max_retries:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(f"  [retry] Gemini call failed ({exc}); retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(delay)


SYSTEM_PROMPT = f"""You are an SRE incident investigator. You have three tools:
fetch_logs, fetch_metrics (each takes a service + time window), and
search_runbooks (takes a free-text symptom query).

Known services: {", ".join(KNOWN_SERVICES)}.

Investigate the alert below: call fetch_logs/fetch_metrics for the service(s)
the alert mentions first, but the true root cause is often an *earlier*
failure in a different, upstream service - if the evidence suggests that,
check other known services too before concluding. Once you've spotted a
clear symptom pattern, call search_runbooks to check it against a known
issue and fix. Once you have enough evidence, STOP calling tools and reply
with a plain-language investigation summary covering: what happened, the
order of failures across services (cite timestamps), the most likely root
cause (the earliest-failing service, not necessarily the one that reported
the alert), and the recommended fix from the runbook match.

If fetch_logs/fetch_metrics report no entries for every service you check
within the given time window, do NOT keep searching indefinitely - after
checking at most 2-3 services, stop and reply with a summary stating that
no evidence was found for this alert in this time window, at low
confidence, rather than exhausting your tool-call budget."""

EXTRACTION_PROMPT = """Extract a structured root-cause-analysis report as JSON from the investigation summary below.
Copy every timestamp exactly as it appears in the summary - do not alter the
year, date, or any digit. Respond with ONLY valid JSON, matching this schema exactly:
{{
  "summary": "one or two sentence plain-language summary",
  "timeline": [{{"timestamp": "one full ISO 8601 datetime, e.g. 2026-08-30T09:15:12Z - never a bare time or a range", "service": "service name", "description": "what happened"}}],
  "root_cause": "the most likely root cause, one or two sentences",
  "evidence": ["short evidence citation", "..."],
  "recommended_fix": "the recommended fix",
  "confidence": "low, medium, or high"
}}

Investigation summary:
{summary}"""

_agent_llm = langchain_chat_model(temperature=0.2).bind_tools(TOOLS)
_extractor_llm = langchain_chat_model(temperature=0.0, response_mime_type="application/json")


def _agent_node(state: MessagesState) -> dict:
    response = _invoke_with_retry(lambda: _agent_llm.invoke(state["messages"]))
    return {"messages": [response]}


def _should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "extract"


def _extract_node(state: MessagesState) -> dict:
    summary = state["messages"][-1].text
    response = _invoke_with_retry(lambda: _extractor_llm.invoke([("user", EXTRACTION_PROMPT.format(summary=summary))]))
    return {"messages": [response]}


def build_agent():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("extract", _extract_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "extract": "extract"})
    graph.add_edge("tools", "agent")
    graph.add_edge("extract", END)
    return graph.compile()


_compiled_agent = build_agent()


def _usage_details(message: AIMessage) -> dict:
    """Extract token counts from an AIMessage into Langfuse's usage_details shape."""
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }


def _log_transcript(tracer: IncidentTracer, messages: list) -> int:
    """Log each model turn as a "generation" observation and each tool
    result as a "tool" observation, from the finished message list."""
    pending_calls: dict[str, dict] = {}
    for i, message in enumerate(messages):
        if isinstance(message, AIMessage):
            name = "rca_extraction" if i == len(messages) - 1 else "agent_turn"
            if message.tool_calls:
                for call in message.tool_calls:
                    pending_calls[call["id"]] = call
                output = "decided to call: " + ", ".join(f"{c['name']}({c['args']})" for c in message.tool_calls)
            else:
                output = message.text
            tracer.log_generation(name, output=output, model=MODEL_NAME, usage_details=_usage_details(message))
        elif isinstance(message, ToolMessage):
            call = pending_calls.get(message.tool_call_id)
            tracer.log_tool(call["name"] if call else message.name, output=message.content, input=call["args"] if call else {})
    return tracer.step_count


_INCONCLUSIVE_REPORT = {
    "summary": "The agent could not reach a conclusion within its investigation step budget.",
    "timeline": [],
    "root_cause": "Unable to determine - no clear evidence pattern was found for this alert within the given time window, or the investigation didn't converge in time.",
    "evidence": [],
    "recommended_fix": "Try a narrower alert or a time window matching one of the seeded incidents' data (see the module README).",
    "confidence": "low",
}


def analyze_incident(incident_id: str, alert: str, services: list[str], start_time: str, end_time: str) -> tuple[dict, int, str | None]:
    """Run the agent loop for one incident and extract a structured RCA
    report, logging every step to Langfuse under incident_id. Returns
    (report_dict, step_count, trace_url).

    Falls back to a generic low-confidence report if LangGraph's
    GraphRecursionError fires (e.g. no evidence found anywhere)."""
    tracer = IncidentTracer(incident_id, alert)
    user_prompt = (
        f"ALERT: {alert}\nServices the alert mentions: {services}\nTime window: {start_time} to {end_time}\n\n"
        "Investigate this incident."
    )
    try:
        final_state = _compiled_agent.invoke(
            {"messages": [("system", SYSTEM_PROMPT), ("user", user_prompt)]},
            config={"recursion_limit": 25},
        )
        step_count = _log_transcript(tracer, final_state["messages"])
        data = json.loads(final_state["messages"][-1].text)
    except GraphRecursionError:
        tracer.log_generation(
            "agent_incomplete",
            output=_INCONCLUSIVE_REPORT["root_cause"],
            model=MODEL_NAME,
            usage_details={"input": 0, "output": 0, "total": 0},
        )
        step_count = tracer.step_count
        data = dict(_INCONCLUSIVE_REPORT)

    data["incident_id"] = incident_id
    tracer.finish(output=data)
    return data, step_count, tracer.get_trace_url()
