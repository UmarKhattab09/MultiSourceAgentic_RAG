"""Run the eval set once through the agent (model + retrieval mode from config.yaml) with LangSmith tracing.

Needs LANGSMITH_API_KEY and LANGSMITH_TRACING=true in .env. Traces land in the LANGSMITH_PROJECT project,
tagged "eval" with the expected tool/answer as metadata. Does not touch eval/results.csv.

Usage: python eval/trace_eval.py
"""
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.checkpoint.memory import MemorySaver
from agent import build_graph  # also loads .env
from llm import load_config

def main():
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true" or not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("Tracing is off: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env")
    cfg = load_config()
    graph = build_graph(MemorySaver())  # in-memory: eval questions must not share chat history
    questions = [json.loads(line) for line in open(ROOT / "eval" / "questions.jsonl", encoding="utf-8")]
    correct = 0
    for i, q in enumerate(questions, 1):
        out = graph.invoke({"messages": [HumanMessage(q["question"])]},
                           {"configurable": {"thread_id": str(uuid.uuid4())}, "run_name": "agent",
                            "tags": ["eval", cfg["model"], cfg["retrieval_mode"]],
                            "metadata": {"expected_tool": q["tool"], "expected_answer": q["expected_answer"]}})
        correct += out["tool"] == q["tool"]
        mark = "OK " if out["tool"] == q["tool"] else "BAD"
        print(f"[{i}/{len(questions)}] {mark} {out['tool']:13} {q['question'][:60]}", flush=True)
    wait_for_all_tracers()  # flush traces before exit
    print(f"done: {correct}/{len(questions)} correct tool | model={cfg['model']} mode={cfg['retrieval_mode']} "
          f"| project={os.getenv('LANGSMITH_PROJECT', 'default')}")

if __name__ == "__main__":
    main()
