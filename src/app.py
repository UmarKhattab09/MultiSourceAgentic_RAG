"""Streamlit chat UI for the agent, with employee sign-in and an HR review page. Run: streamlit run src/app.py"""
import sys
import ollama
import streamlit as st
from langchain_core.messages import HumanMessage
from agent import NUM_CTX, build_graph, context_label, final_state, pending_action, step_label, stream_answer
from embeddings import OLLAMA_URL, ROOT
from llm import load_config

sys.path.insert(0, str(ROOT / "mcp_servers"))
import hr_server

st.set_page_config(page_title="Multi-Source RAG Agent", page_icon="🔎")

@st.cache_resource
def get_graph():
    return build_graph()  # SQLite chat memory, shared with src/agent.py

user = st.session_state.get("user")
if not user:
    st.title("🔎 Sign in")
    with st.form("login"):
        employee_id = st.text_input("Employee id", placeholder="E001")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if found := hr_server.login(employee_id, password):
                st.session_state.user = found
                st.rerun()
            st.error("Wrong employee id or password.")
    st.stop()

graph = get_graph()
config = load_config()
prefix = f"{user['id']}/"  # chat sessions belong to the signed-in employee

with st.sidebar:
    st.write(f"Signed in as **{user['name']}** ({user['id']}, {user['department']})")
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    page = st.radio("Page", ["Chat", "My leave requests", *(["HR: pending requests"] if user["is_hr"] else [])])

    st.header("Session")
    sessions = list(dict.fromkeys(c.config["configurable"]["thread_id"].removeprefix(prefix)
                                  for c in graph.checkpointer.list(None) if c.config["configurable"]["thread_id"].startswith(prefix)))
    picked = st.selectbox("Saved sessions", ["(new session)", *sessions])
    thread = prefix + (st.text_input("Session id", value="" if picked == "(new session)" else picked) or "default")
    state = graph.get_state({"configurable": {"thread_id": thread}}).values
    tokens = state.get("context_tokens") or 0
    st.progress(min(tokens / NUM_CTX, 1.0), text=f"Context window: {context_label(tokens)}")
    st.caption("Largest prompt + reply of the last question. Past 100%, Ollama drops the oldest text, instructions first.")

    st.header("Settings")
    try:
        installed = [m.model for m in ollama.Client(host=OLLAMA_URL).list().models if "embed" not in m.model]
    except Exception:
        installed = []
    models = list(dict.fromkeys([config["model"], *installed]))
    model = st.selectbox("Model", models)
    modes = ["dense", "hybrid", "hybrid_rerank"]
    mode = st.selectbox("Retrieval mode", modes, index=modes.index(config["retrieval_mode"]))

if page == "My leave requests":
    st.title("My leave requests")
    st.write(hr_server.leave_balance(user["id"])["balances"])
    st.dataframe(hr_server.my_requests(user["id"]), hide_index=True,
                 column_order=["id", "leave_type", "start_date", "end_date", "days", "reason", "status", "created_at"])
    st.stop()

if page == "HR: pending requests":
    st.title("Pending leave requests")
    rows = hr_server.pending_requests(user["id"])
    if not rows:
        st.info("No pending requests.")
    for r in rows:
        info, approve, reject = st.columns([4, 1, 1])
        info.markdown(f"**#{r['id']} {r['name']}** ({r['department']}): {r['leave_type']}, {r['start_date']} to {r['end_date']}, "
                      f"{r['days']} day(s). {r['reason'] or ''}")
        for col, label, ok in ((approve, "Approve", True), (reject, "Reject", False)):
            if col.button(label, key=f"{label}{r['id']}", type="primary" if ok else "secondary"):
                try:
                    hr_server.decide_request(user["id"], r["id"], approve=ok)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
    st.stop()

st.title("🔎 Multi-Source RAG Agent")
st.caption("Ask about the company documents, do math, search the web, ask the date, check or book leave, or use Jira.")

for m in state.get("messages", []):
    st.chat_message("user" if isinstance(m, HumanMessage) else "assistant").markdown(m.content)

def run(**kwargs):
    """Ask or resume, streaming the answer, then rerun so the history above shows the result; a paused write action waits for the buttons."""
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                st.write_stream(stream_answer(graph, thread_id=thread, model=model, retrieval_mode=mode, employee_id=user["id"], **kwargs))
        except Exception as e:
            st.error(f"The agent failed: {e}. Is Ollama running at {OLLAMA_URL}?")
            st.stop()
    out = final_state(graph, thread)
    st.session_state.pending = pending_action(out)
    st.session_state.last = None if st.session_state.pending else out
    st.rerun()

if last := st.session_state.get("last"):
    st.caption("Tools: " + (", ".join(f"`{step_label(s)}`" for s in last["steps"]) or "none")
               + f" · router attempts: {last['attempts']}")
    if last["tool"] == "search_docs":
        with st.expander("Retrieved chunks"):
            st.text(last["tool_output"])

if action := st.session_state.get("pending"):
    with st.chat_message("assistant"):
        st.warning(f"Confirm **{action['tool']}** before it runs:")
        st.json(action["args"])
        approve, decline = st.columns(2)
        if approve.button("Approve", type="primary"):
            run(resume=True)
        if decline.button("Decline"):
            run(resume=False)
elif question := st.chat_input("Ask a question"):
    st.chat_message("user").markdown(question)
    run(question=question)
