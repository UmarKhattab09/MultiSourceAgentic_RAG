"""LangGraph agent: router (model picks a tool as JSON) -> tool -> answer, with SQLite chat memory."""
import json
import sqlite3
import sys
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from embeddings import OLLAMA_URL, ROOT
from llm import load_config
from tools import TOOLS

load_dotenv(ROOT / ".env")  # LangSmith tracing turns on when LANGSMITH_TRACING + LANGSMITH_API_KEY are set there

MAX_ATTEMPTS = 3
HISTORY = 6  # recent messages the router and answerer see

ROUTER_PROMPT = """You are a router. Pick exactly one tool for the user's latest message.
Tools:
- search_docs: questions about the company documents: anything about Zava (its history, founding, milestones, values, vacation tiers, jobs, policies), employee handbook, Northwind health plans and benefits, PerksPlus, role library, the 2023 financial market analysis report. Zava is the user's own company: never use web_search for Zava.
- calculator: arithmetic. The input must be a plain math expression like "(1200*0.15)+30".
- web_search: current events, news, or general facts not in the company documents.
- get_datetime: the current date, time, or day of week.
- none: greetings, small talk, or questions about this conversation itself.

The input is the user's own question rewritten so it stands alone (resolve "it", "that", "and for X?" from the conversation). Never copy the tool descriptions above into the input.

Examples:
User: What does PerksPlus reimburse? -> {"tool": "search_docs", "input": "What does PerksPlus reimburse?"}
User (after asking about the Standard plan deductible): And for Plus? -> {"tool": "search_docs", "input": "What is the deductible for the Northwind Health Plus plan?"}
User: What is 15% of 84,000? -> {"tool": "calculator", "input": "0.15*84000"}
User: Who is the CEO of Microsoft? -> {"tool": "web_search", "input": "current CEO of Microsoft"}
User: What's today's date? -> {"tool": "get_datetime", "input": ""}
User: Thanks! -> {"tool": "none", "input": ""}

Reply with ONLY a JSON object, no other text: {"tool": "<tool name>", "input": "<tool input>"}"""

ANSWER_PROMPT = """Answer the user's latest message using the tool result below.
If the tool was search_docs, answer ONLY from that result, cite every fact as [Source: file, page N] copied from the result, and say "I do not know based on the provided documents." if it is not there.
Be concise.

Tool: {tool}
Tool result:
{result}"""

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool: str
    tool_input: str
    tool_output: str
    attempts: int       # router calls needed to get valid JSON
    json_valid: bool    # did the first router reply parse?

def parse_route(text: str) -> dict:
    """Strict parse: JSON object (optionally in a ``` fence) with a known tool. Raises on anything else."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    route = json.loads(text)
    if not isinstance(route, dict) or route.get("tool") not in [*TOOLS, "none"]:
        raise ValueError(f"tool must be one of {[*TOOLS, 'none']}")
    route["input"] = str(route.get("input", ""))
    return route

def llm(config) -> ChatOllama:
    model = config["configurable"].get("model") or load_config()["model"]
    # Cap output and time: small models can loop forever on a reply (phi4-mini did, hanging the eval)
    return ChatOllama(model=model, base_url=OLLAMA_URL, temperature=0, reasoning=False, num_predict=512, client_kwargs={"timeout": 180})

def router(state: State, config) -> dict:
    messages = [SystemMessage(ROUTER_PROMPT), *state["messages"][-HISTORY:]]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = llm(config).invoke(messages).content
        try:
            route = parse_route(reply)
            return {"tool": route["tool"], "tool_input": route["input"], "attempts": attempt, "json_valid": attempt == 1}
        except (ValueError, json.JSONDecodeError) as e:
            messages += [AIMessage(reply), HumanMessage(f"Invalid reply ({e}). Reply with ONLY the JSON object.")]
    # Give up on routing: the documents are the most likely target
    return {"tool": "search_docs", "tool_input": state["messages"][-1].content, "attempts": MAX_ATTEMPTS, "json_valid": False}

def run_tool(state: State, config) -> dict:
    if state["tool"] == "none":
        return {"tool_output": "(no tool used)"}
    if state["tool"] == "search_docs":
        return {"tool_output": TOOLS["search_docs"](state["tool_input"], config["configurable"].get("retrieval_mode"))}
    return {"tool_output": TOOLS[state["tool"]](state["tool_input"])}

def answer(state: State, config) -> dict:
    system = ANSWER_PROMPT.format(tool=state["tool"], result=state["tool_output"])
    reply = llm(config).invoke([SystemMessage(system), *state["messages"][-HISTORY:]])
    return {"messages": [AIMessage(reply.content)]}

def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("tool", run_tool)
    g.add_node("answer", answer)
    g.add_edge(START, "router")
    g.add_edge("router", "tool")
    g.add_edge("tool", "answer")
    g.add_edge("answer", END)
    if checkpointer is None:
        checkpointer = SqliteSaver(sqlite3.connect(ROOT / "chat_memory.db", check_same_thread=False))
    return g.compile(checkpointer=checkpointer)

def ask(graph, question: str, thread_id: str = "default", model: str | None = None, retrieval_mode: str | None = None) -> dict:
    config = {"configurable": {"thread_id": thread_id, "model": model, "retrieval_mode": retrieval_mode}}
    return graph.invoke({"messages": [HumanMessage(question)]}, config)

if __name__ == "__main__":
    assert parse_route('```json\n{"tool": "calculator", "input": "2+2"}\n```')["tool"] == "calculator"
    for bad in ['not json', '{"tool": "hack"}', '[1]']:
        try:
            parse_route(bad); raise AssertionError(bad)
        except (ValueError, json.JSONDecodeError):
            pass

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252 and crash on emoji in replies
    graph = build_graph()
    sessions = dict.fromkeys(c.config["configurable"]["thread_id"] for c in graph.checkpointer.list(None))
    print("Saved sessions:", ", ".join(sessions) or "(none)")
    thread = input("Session id (blank = default): ").strip() or "default"
    for m in graph.get_state({"configurable": {"thread_id": thread}}).values.get("messages", []):
        print(f"{'You' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}")
    while (q := input("\nYou: ").strip()) not in ("", "exit", "quit"):
        out = ask(graph, q, thread)
        print(f"[{out['tool']}: {out['tool_input']}]")
        print("Agent:", out["messages"][-1].content)
