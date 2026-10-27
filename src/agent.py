"""LangGraph agent: router (model picks a tool as JSON) -> tool -> router ... -> answer, with SQLite chat memory.

Tools: the built-ins in tools.py plus MCP tools (Zava HR, Jira). Write actions pause for human confirmation
(LangGraph interrupt); resume with ask(..., resume=True/False). Prompt-injection defenses live in defenses.py.
"""
import json
import os
import re
import sqlite3
import sys
from datetime import date
from getpass import getpass
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from dotenv import load_dotenv
from embeddings import OLLAMA_URL, ROOT
from llm import load_config
import defenses
import mcp_tools
from tools import TOOLS

load_dotenv(ROOT / ".env")  # LangSmith tracing and Jira credentials

MAX_ATTEMPTS = 3
MAX_STEPS = 3  # tool calls per question
HISTORY = 6  # recent messages the router and answerer see

BUILTIN_TOOLS = {
    "search_docs": "questions about the company documents: anything about Zava (its history, founding, milestones, values, vacation tiers, jobs, policies), employee handbook, Northwind health plans and benefits, PerksPlus, role library, the 2023 financial market analysis report. Zava is the user's own company: never use web_search for Zava.",
    "calculator": 'arithmetic. The input must be a plain math expression like "(1200*0.15)+30".',
    "web_search": "current events, news, or general facts not in the company documents.",
    "get_datetime": "the current date, time, or day of week.",
}
EXAMPLES = [  # (tool the example needs, example line)
    ("search_docs", 'User: What does PerksPlus reimburse? -> {"tool": "search_docs", "input": "What does PerksPlus reimburse?"}'),
    ("search_docs", 'User (after asking about the Standard plan deductible): And for Plus? -> {"tool": "search_docs", "input": "What is the deductible for the Northwind Health Plus plan?"}'),
    ("calculator", 'User: What is 15% of 84,000? -> {"tool": "calculator", "input": "0.15*84000"}'),
    ("web_search", 'User: Who is the CEO of Microsoft? -> {"tool": "web_search", "input": "current CEO of Microsoft"}'),
    ("get_datetime", 'User: What\'s today\'s date? -> {"tool": "get_datetime", "input": ""}'),
    ("get_leave_balance", 'User: How many vacation days do I have left? -> {"tool": "get_leave_balance", "input": {}}'),
    ("submit_leave_request", 'User: Book vacation from 2026-11-09 to 2026-11-13 -> {"tool": "submit_leave_request", "input": {"leave_type": "vacation", "start_date": "2026-11-09", "end_date": "2026-11-13", "reason": ""}}'),
    ("jira_search", 'User: Show my Jira tickets -> {"tool": "jira_search", "input": {"jql": "ORDER BY created DESC"}}'),
    ("jira_create_issue", 'User: Open an IT ticket, my laptop will not boot -> {"tool": "jira_create_issue", "input": {"project_key": "PROJECT", "summary": "Laptop will not boot", "issue_type": "Task"}}'),
    ("none", 'User: Thanks! -> {"tool": "none", "input": ""}'),
]

ROUTER_PROMPT = """You are a router. Pick exactly one tool for the user's latest message. Today is {today}.
Tools:
{tools}
- none: no tool needed: greetings, small talk, questions about this conversation, or the tool results already answer the user.
{rule}
For search_docs, calculator, web_search and get_datetime the input is a string: the user's own question rewritten so it stands alone (resolve "it", "that", "and for X?" from the conversation). Never copy the tool descriptions above into the input.
For every other tool the input is a JSON object with that tool's arguments.

Examples:
{examples}

Reply with ONLY a JSON object, no other text: {{"tool": "<tool name>", "input": <tool input>}}"""

DOCS_RULE = 'For the search_docs result, answer ONLY from it, cite every fact as [Source: file, page N] copied from the result, and say "I do not know based on the provided documents." if it is not there.'
ANSWER_PROMPT = """Answer the user's latest message using the tool results below.
{docs_rule}
For other tools, report what the result says, e.g. "You have no Jira tickets yet."
If an action was blocked or declined, say so plainly. Be concise.
Write plain sentences for the user: never write tool calls, <tool_code>, tags or raw JSON.
{rule}
{results}"""

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool: str           # first tool picked for the question (what the eval scores)
    tool_input: str
    tool_output: str
    attempts: int       # router calls needed to get valid JSON (first route)
    json_valid: bool    # did the first router reply parse?
    steps: list         # this question's tool calls: {tool, input, output, status}
    next_tool: str
    next_input: object
    context_tokens: int  # largest prompt + reply of this question's LLM calls, as counted by Ollama

def parse_route(text: str, allowed=None, structured=()) -> dict:
    """Strict parse: JSON object (optionally in a ``` fence) with a known tool. Raises on anything else.
    Tools in `structured` take an object of arguments; the others take a string."""
    allowed = allowed or [*TOOLS, "none"]
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    route = json.loads(text)
    if not isinstance(route, dict) or route.get("tool") not in allowed:
        raise ValueError(f"tool must be one of {allowed}")
    if route["tool"] in structured:
        if not isinstance(route.get("input", {}), dict):
            raise ValueError(f"input for {route['tool']} must be a JSON object of its arguments")
        route["input"] = route.get("input") or {}
    else:
        route["input"] = str(route.get("input", ""))
    return route

def settings(config) -> dict:
    c = config["configurable"]
    cfg = load_config()
    return {"model": c.get("model") or cfg["model"], "retrieval_mode": c.get("retrieval_mode"),
            "employee_id": c.get("employee_id"),
            "defenses": cfg.get("defenses", True) if c.get("defenses") is None else c["defenses"]}

NUM_CTX = 4096  # Ollama's default on this 4 GB GPU; set explicitly so the context usage shown is real

def used_tokens(state: State, reply) -> int:
    return max(state.get("context_tokens") or 0, (reply.usage_metadata or {}).get("total_tokens", 0))

def context_label(tokens: int) -> str:
    return f"{tokens:,} / {NUM_CTX:,} tokens ({tokens / NUM_CTX:.0%})"

def step_label(s: dict) -> str:
    """Tool, status and how much of the context window its output takes."""
    tokens = len(str(s["output"])) // 4  # shortcut: ~4 chars per token estimate, use the model tokenizer if exact counts matter
    return f"{s['tool']} ({s['status']}, ~{tokens} tok = {tokens / NUM_CTX:.0%} of {NUM_CTX} context)"

def llm(config) -> ChatOllama:
    # Cap output and time: small models can loop forever on a reply (phi4-mini did, hanging the eval)
    return ChatOllama(model=settings(config)["model"], base_url=OLLAMA_URL, temperature=0, reasoning=False,
                      num_predict=512, num_ctx=NUM_CTX, client_kwargs={"timeout": 180})

def mcp_specs(config) -> dict:
    return mcp_tools.tool_specs(session_bound=settings(config)["defenses"])

def router_prompt(specs: dict, defended: bool) -> str:
    lines = [f"- {name}: {desc}" for name, desc in BUILTIN_TOOLS.items()]
    for name, spec in specs.items():
        args = ", ".join(spec["schema"].get("properties", {}))
        confirm = " Asks the user to confirm first." if name in mcp_tools.WRITE_TOOLS else ""
        if name == "jira_search":
            confirm = " Results are already limited to the signed-in employee's own tickets: never filter by assignee or reporter."
        lines.append(f"- {name}({args}): {spec['description'].strip().splitlines()[0]}{confirm}")
    available = {*BUILTIN_TOOLS, *specs, "none"}
    examples = "\n".join(line for tool, line in EXAMPLES if tool in available).replace('"PROJECT"', json.dumps(os.getenv("JIRA_PROJECT", "IT")))
    return ROUTER_PROMPT.format(today=date.today().strftime("%A %Y-%m-%d"), tools="\n".join(lines), rule=defenses.UNTRUSTED_RULE if defended else "", examples=examples)

def format_results(steps: list, defended: bool) -> str:
    blocks = []
    for s in steps:
        output = defenses.wrap(s["output"]) if defended else s["output"]
        blocks.append(f"Tool: {s['tool']} ({s['status']})\nInput: {json.dumps(s['input'])}\nResult:\n{output}")
    return "\n\n".join(blocks)

def user_text(state: State) -> str:
    return "\n".join(m.content for m in state["messages"][-HISTORY:] if isinstance(m, HumanMessage))

def router(state: State, config) -> dict:
    steps = state.get("steps") or []
    if len(steps) >= MAX_STEPS or (steps and steps[-1]["status"] in ("declined", "blocked")):
        return {"next_tool": "none"}  # after a "no" (user or defense) answer, never ask again with reworded args
    defended, specs = settings(config)["defenses"], mcp_specs(config)
    messages = [SystemMessage(router_prompt(specs, defended)), *state["messages"][-HISTORY:]]
    if steps:
        messages.append(HumanMessage(f"Tool results so far:\n\n{format_results(steps, defended)}\n\n"
                                     'If these answer my latest message, reply {"tool": "none", "input": ""}. Otherwise pick the next tool.'))
    allowed = [*BUILTIN_TOOLS, *specs, "none"]
    parsed, tokens = False, state.get("context_tokens") or 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        msg = llm(config).invoke(messages)
        reply, tokens = msg.content, used_tokens({"context_tokens": tokens}, msg)
        try:
            route, parsed = parse_route(reply, allowed, specs), True
            break
        except (ValueError, json.JSONDecodeError) as e:
            messages += [AIMessage(reply), HumanMessage(f"Invalid reply ({e}). Reply with ONLY the JSON object.")]
    if not parsed:
        # Give up on routing: on the first step the documents are the most likely target, later just answer
        route = {"tool": "search_docs" if not steps else "none", "input": state["messages"][-1].content}
    if any(s["tool"] == route["tool"] and s["input"] == route["input"] for s in steps):
        route = {"tool": "none", "input": ""}  # repeating a call cannot add anything: answer with what we have
    update = {"next_tool": route["tool"], "next_input": route["input"], "context_tokens": tokens}
    if not steps:
        update |= {"tool": route["tool"], "attempts": attempt, "json_valid": parsed and attempt == 1,
                   "tool_input": route["input"] if isinstance(route["input"], str) else json.dumps(route["input"]),
                   "tool_output": "(no tool used)"}
    return update

def execute(name: str, args, s: dict) -> str:
    if name == "search_docs":
        return TOOLS["search_docs"](str(args), s["retrieval_mode"])
    if name in TOOLS:
        return TOOLS[name](str(args))
    return mcp_tools.call_tool(name, args, s["employee_id"], session_bound=s["defenses"])

def run_tool(state: State, config) -> dict:
    s = settings(config)
    name, args = state["next_tool"], state["next_input"]
    steps = list(state.get("steps") or [])
    status, output = "executed", None
    if s["defenses"]:
        output = defenses.check_call(name, args, user_text(state), [step["output"] for step in steps])
        status = "blocked" if output else status
    if status == "executed" and name == "jira_create_issue" and s["employee_id"]:
        args = mcp_tools.scope_jira(name, args, s["employee_id"])  # the user approves exactly what will be sent
    if status == "executed" and name in mcp_tools.WRITE_TOOLS and not interrupt({"tool": name, "args": args}):
        status, output = "declined", "The user declined this action."
    if output is None:
        output = execute(name, args, s)
    steps.append({"tool": name, "input": args, "output": output, "status": status})
    return {"steps": steps, **({"tool_output": output} if len(steps) == 1 else {})}

def clean_answer(text: str) -> str:
    """Backstop for small models (gemma4:e2b) that echo tool calls and data tags into the reply."""
    text = re.sub(r"<tool_code>.*?</tool_code>", "", text, flags=re.S)
    return re.sub(r"</?\s*untrusted_\s*data\s*>", "", text).strip()

def answer(state: State, config) -> dict:
    steps = state.get("steps") or []
    defended = settings(config)["defenses"]
    results = format_results(steps, defended) if steps else "(no tool used)"
    used_docs = any(s["tool"] == "search_docs" for s in steps)  # small models apply the docs rule to every tool otherwise
    system = ANSWER_PROMPT.format(docs_rule=DOCS_RULE if used_docs else "", rule=defenses.UNTRUSTED_RULE if defended and steps else "", results=results)
    reply = llm(config).invoke([SystemMessage(system), *state["messages"][-HISTORY:]])
    return {"messages": [AIMessage(clean_answer(reply.content))], "context_tokens": used_tokens(state, reply)}

def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("tool", run_tool)
    g.add_node("answer", answer)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda state: "answer" if state["next_tool"] == "none" else "tool")
    g.add_edge("tool", "router")
    g.add_edge("answer", END)
    if checkpointer is None:
        checkpointer = SqliteSaver(sqlite3.connect(ROOT / "chat_memory.db", check_same_thread=False))
    return g.compile(checkpointer=checkpointer)

def ask(graph, question: str = "", thread_id: str = "default", model: str | None = None, retrieval_mode: str | None = None,
        employee_id: str | None = None, defenses: bool | None = None, resume: bool | None = None) -> dict:
    """Ask a question, or resume a paused write action with resume=True (approve) / False (decline).
    A result containing "__interrupt__" is waiting for that decision."""
    return graph.invoke(*run_args(question, thread_id, model, retrieval_mode, employee_id, defenses, resume))

def stream_answer(graph, question: str = "", thread_id: str = "default", model: str | None = None, retrieval_mode: str | None = None,
                  employee_id: str | None = None, defenses: bool | None = None, resume: bool | None = None):
    """Same as ask(), but yields the answer's text as the model writes it. Read the result with final_state() afterwards."""
    for chunk, meta in graph.stream(*run_args(question, thread_id, model, retrieval_mode, employee_id, defenses, resume),
                                    stream_mode="messages"):
        if meta.get("langgraph_node") == "answer" and isinstance(chunk, AIMessageChunk) and chunk.content:
            yield chunk.content

def final_state(graph, thread_id: str) -> dict:
    """What ask() would have returned for the last run on this thread, including a pending "__interrupt__"."""
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    interrupts = [i for task in snap.tasks for i in task.interrupts]
    return {**snap.values, **({"__interrupt__": interrupts} if interrupts else {})}

def run_args(question, thread_id, model, retrieval_mode, employee_id, defenses, resume) -> tuple:
    config = {"configurable": {"thread_id": thread_id, "model": model, "retrieval_mode": retrieval_mode,
                               "employee_id": employee_id, "defenses": defenses}}
    payload = Command(resume=resume) if resume is not None else {"messages": [HumanMessage(question)], "steps": [], "context_tokens": 0}
    return payload, config

def pending_action(out: dict) -> dict | None:
    """The write action waiting for confirmation, if any: {"tool", "args"}."""
    return out["__interrupt__"][0].value if out.get("__interrupt__") else None

if __name__ == "__main__":
    assert parse_route('```json\n{"tool": "calculator", "input": "2+2"}\n```')["tool"] == "calculator"
    for bad in ['not json', '{"tool": "hack"}', '[1]']:
        try:
            parse_route(bad); raise AssertionError(bad)
        except (ValueError, json.JSONDecodeError):
            pass

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252 and crash on emoji in replies
    sys.path.insert(0, str(ROOT / "mcp_servers"))
    import hr_server
    if not (user := hr_server.login(input("Employee id: "), getpass("Password: "))):
        sys.exit("Wrong employee id or password.")
    employee = user["id"]
    graph = build_graph()
    prefix = f"{employee}/"  # sessions belong to the signed-in employee
    sessions = dict.fromkeys(c.config["configurable"]["thread_id"].removeprefix(prefix)
                             for c in graph.checkpointer.list(None) if c.config["configurable"]["thread_id"].startswith(prefix))
    print(f"Hi {user['name']}. Saved sessions:", ", ".join(sessions) or "(none)")
    thread = prefix + (input("Session id (blank = default): ").strip() or "default")
    for m in graph.get_state({"configurable": {"thread_id": thread}}).values.get("messages", []):
        print(f"{'You' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}")
    while (q := input("\nYou: ").strip()) not in ("", "exit", "quit"):
        print("Agent: ", end="", flush=True)
        for text in stream_answer(graph, q, thread, employee_id=employee):
            print(text, end="", flush=True)
        while action := pending_action(out := final_state(graph, thread)):
            print(f"\nConfirm {action['tool']} {json.dumps(action['args'])}?")
            approved = input("Approve? [y/N] ").strip().lower() == "y"
            print("Agent: ", end="", flush=True)
            for text in stream_answer(graph, thread_id=thread, employee_id=employee, resume=approved):
                print(text, end="", flush=True)
        print("\n[" + ", ".join(step_label(s) for s in out.get("steps", [])) + "]")
        print("Context:", context_label(out.get("context_tokens") or 0))
