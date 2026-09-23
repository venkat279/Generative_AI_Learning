"""Module 10 mini application: a multi-agent pipeline built with
LangGraph over Google's Gemini API (gemini-3.1-flash-lite), with a
human-in-the-loop checkpoint.

Agent roles (Section 2) wired in a Pipeline communication pattern
(Section 3):

    researcher -> writer -> human_review (pauses here) -> finalize

The human_review step (Section 6) pauses the whole graph with
langgraph.types.interrupt(), shows you the drafted paragraph, and
waits for a real decision - approve it, reject it, or type replacement
text - before the graph resumes and finalizes.

Run:
    python app.py
"""

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import langchain_chat_model  # noqa: E402

llm = langchain_chat_model(temperature=0.4)


class PipelineState(TypedDict):
    topic: str
    notes: str
    draft: str
    decision: str
    final: str


def researcher_node(state: PipelineState) -> dict:
    """Researcher agent: gathers the key points to write about."""
    response = llm.invoke(
        "You are a research agent. In 2-3 short bullet points, list the key "
        f"facts someone would need to write a short paragraph about: {state['topic']}"
    )
    return {"notes": response.text}


def writer_node(state: PipelineState) -> dict:
    """Writer agent: turns the researcher's notes into a short paragraph."""
    response = llm.invoke(
        "You are a writer agent. Using ONLY these research notes, write one "
        f"short, engaging paragraph (3-4 sentences).\n\nNotes:\n{state['notes']}"
    )
    return {"draft": response.text}


def human_review_node(state: PipelineState) -> dict:
    """Pauses the graph and waits for a human decision on the draft."""
    decision = interrupt(
        {
            "question": "Approve this draft, reject it, or type replacement text.",
            "topic": state["topic"],
            "draft": state["draft"],
        }
    )
    return {"decision": decision}


def finalize_node(state: PipelineState) -> dict:
    """Applies the human's decision to produce the final output."""
    decision = state["decision"].strip()
    lowered = decision.lower()
    if lowered in ("", "approve", "approved", "yes", "y"):
        final = state["draft"]
    elif lowered in ("reject", "rejected", "no", "n"):
        final = "[rejected by human reviewer - no final output]"
    else:
        final = decision  # anything else is treated as edited replacement text
    return {"final": final}


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())


def run(pipeline, topic: str) -> None:
    print(f"\n{'=' * 70}\nTopic: {topic}")
    config = {"configurable": {"thread_id": topic}}

    result = pipeline.invoke({"topic": topic}, config=config)
    print(f"\n[researcher] notes:\n{result['notes']}")
    print(f"\n[writer] draft:\n{result['draft']}")

    decision = input(
        "\n[human] Approve, reject, or type replacement text, then press Enter: "
    ).strip()

    final_state = pipeline.invoke(Command(resume=decision), config=config)
    print(f"\n[finalize] final output:\n{final_state['final']}")


TOPICS = [
    "why code review matters",
]


def main() -> None:
    pipeline = build_pipeline()
    for topic in TOPICS:
        run(pipeline, topic)


if __name__ == "__main__":
    main()
