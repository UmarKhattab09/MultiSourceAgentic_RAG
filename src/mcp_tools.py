"""MCP tools for the agent: the Zava HR server (always) and Jira via mcp-atlassian (when JIRA_URL is set in .env)."""
import asyncio
import json
import os
import re
import sys
from functools import lru_cache
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from embeddings import ROOT

WRITE_TOOLS = {"submit_leave_request", "jira_create_issue"}  # always need the user's confirmation
JIRA_TOOLS = "jira_create_issue,jira_get_issue,jira_search"  # the only mcp-atlassian tools loaded
ISSUE_TYPES = ("Task", "Incident", "Service Request")  # shortcut: the KAN project's types, read Jira's createmeta if the project changes

def servers(employee_id: str | None = None, session_bound: bool = True) -> dict:
    """Connection configs. The employee id goes to the HR server process, never through the model."""
    hr_env = {"HR_INSECURE_EMPLOYEE_ID": "0" if session_bound else "1"}
    for key, value in (("ZAVA_EMPLOYEE_ID", employee_id), ("HR_DB", os.getenv("HR_DB"))):
        if value:
            hr_env[key] = value
    config = {"hr": {"transport": "stdio", "command": sys.executable,
                     "args": [str(ROOT / "mcp_servers" / "hr_server.py")], "env": hr_env}}
    if os.getenv("JIRA_URL"):
        config["jira"] = {"transport": "stdio", "command": "uvx", "args": ["mcp-atlassian==0.23.1"],
                          "env": {"JIRA_URL": os.environ["JIRA_URL"], "JIRA_USERNAME": os.getenv("JIRA_USERNAME", ""),
                                  "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN", ""), "ENABLED_TOOLS": JIRA_TOOLS,
                                  "JIRA_PROJECTS_FILTER": os.getenv("JIRA_PROJECT", "IT")}}
    return config

def first_error(e: BaseException) -> BaseException:
    while isinstance(e, BaseExceptionGroup):  # the MCP client wraps errors in task-group exceptions
        e = e.exceptions[0]
    return e

@lru_cache
def tool_specs(session_bound: bool = True) -> dict[str, dict]:
    """{tool name: {server, description, schema}} for every reachable server. Listed once per process."""
    async def list_tools():
        specs = {}
        for name, connection in servers(session_bound=session_bound).items():
            try:
                for tool in await MultiServerMCPClient({name: connection}).get_tools():
                    specs[tool.name] = {"server": name, "description": tool.description, "schema": tool.args_schema}
            except Exception as e:
                print(f"MCP server '{name}' unavailable, its tools are disabled: {first_error(e)}", file=sys.stderr)
        return specs
    return asyncio.run(list_tools())

def jira_label(employee_id: str) -> str:
    return f"zava-{employee_id.lower()}"

def scope_jira(name: str, args: dict, employee_id: str | None) -> dict:
    """All employees share one Jira account, so tickets are tagged and filtered by the session's employee here, never by the model."""
    if not employee_id:
        raise ValueError("no employee signed in")
    args, label = dict(args), jira_label(employee_id)
    if name == "jira_create_issue":
        args = {k: v for k, v in args.items() if v not in (None, "", [], {})}  # small models fill every optional arg with blanks
        args["project_key"] = os.getenv("JIRA_PROJECT", "IT")  # the project comes from config, never the model
        args.pop("assignee", None)  # the model must not assign people (it invents values like "currentUser()")
        if args.get("issue_type") not in ISSUE_TYPES:
            args["issue_type"] = "Task"
        extra = args.get("additional_fields") or {}
        extra = json.loads(extra) if isinstance(extra, str) else dict(extra)
        extra["labels"] = [*(l for l in extra.get("labels", []) if l != label), label]  # idempotent: scoped for the approval card, then again on the call
        args["additional_fields"] = json.dumps(extra)
        reported = f"Reported by Zava employee {employee_id}."
        if not (args.get("description") or "").startswith(reported):
            args["description"] = f"{reported}\n\n{args.get('description') or ''}".strip()
    elif name == "jira_search":
        jql = args.get("jql") or ""
        m = re.search(r"\border\s+by\b", jql, re.I)
        where, order = (jql[:m.start()], jql[m.end():]) if m else (jql, "")
        unquoted, depth = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', "", where), 0
        for ch in unquoted:  # the model's clause must stay inside our parentheses
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if depth < 0:
                raise ValueError("unbalanced parentheses in JQL")
        if depth or '"' in unquoted or "'" in unquoted:
            raise ValueError("unbalanced parentheses or quotes in JQL")
        args["jql"] = f'labels = "{label}"' + (f" AND ({where.strip()})" if where.strip() else "") + f" ORDER BY {order.strip() or 'created DESC'}"
    elif name == "jira_get_issue":
        args["fields"] = "summary,status,issuetype,priority,assignee,reporter,created,updated,description,labels"
    return args

def owns_issue(text: str, employee_id: str) -> bool:
    try:
        return jira_label(employee_id) in json.loads(text).get("labels", [])
    except (ValueError, AttributeError):
        return False

def call_tool(name: str, args: dict, employee_id: str | None = None, session_bound: bool = True) -> str:
    """Run one MCP tool in a fresh server session and return its text result (errors come back as text too)."""
    if name.startswith("jira_"):
        try:
            args = scope_jira(name, args, employee_id)
        except ValueError as e:
            return f"Error: {e}"
    text = run_mcp(name, args, employee_id, session_bound)
    if name == "jira_get_issue" and not owns_issue(text, employee_id):
        return "Error: issue not found among your tickets."
    return text

def run_mcp(name: str, args: dict, employee_id: str | None, session_bound: bool) -> str:
    server = tool_specs(session_bound=session_bound)[name]["server"]  # outside run(): tool_specs starts its own event loop
    async def run():
        client = MultiServerMCPClient(servers(employee_id, session_bound))
        async with client.session(server) as session:
            tools = {tool.name: tool for tool in await load_mcp_tools(session)}
            return await tools[name].ainvoke(args)
    try:
        result = asyncio.run(run())
    except Exception as e:
        return f"Error: {first_error(e)}"
    if isinstance(result, list):  # MCP content blocks
        return "\n".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in result)
    return result if isinstance(result, str) else json.dumps(result)
