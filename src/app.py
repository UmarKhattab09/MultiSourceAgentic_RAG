"""Streamlit chat UI for the agent. Run: streamlit run src/app.py"""
import ollama
import streamlit as st
from langchain_core.messages import HumanMessage
from agent import ask, build_graph
from embeddings import OLLAMA_URL
from llm import load_config

st.set_page_config(page_title="Multi-Source RAG Agent", page_icon="🔎")

@st.cache_resource
def get_graph():
    return build_graph()  # SQLite chat memory, shared with src/agent.py

graph = get_graph()
config = load_config()

with st.sidebar:
    st.header("Session")
    sessions = list(dict.fromkeys(c.config["configurable"]["thread_id"] for c in graph.checkpointer.list(None)))
    picked = st.selectbox("Saved sessions", ["(new session)", *sessions])
    thread = st.text_input("Session id", value="" if picked == "(new session)" else picked) or "default"

    st.header("Settings")
    try:
        installed = [m.model for m in ollama.Client(host=OLLAMA_URL).list().models if "embed" not in m.model]
    except Exception:
        installed = []
    models = list(dict.fromkeys([config["model"], *installed]))
    model = st.selectbox("Model", models)
    modes = ["dense", "hybrid", "hybrid_rerank"]
    mode = st.selectbox("Retrieval mode", modes, index=modes.index(config["retrieval_mode"]))

st.title("🔎 Multi-Source RAG Agent")
st.caption("Ask about the company documents, do math, search the web, or ask the date.")

state = graph.get_state({"configurable": {"thread_id": thread}}).values
for m in state.get("messages", []):
    st.chat_message("user" if isinstance(m, HumanMessage) else "assistant").markdown(m.content)

if question := st.chat_input("Ask a question"):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                out = ask(graph, question, thread, model, mode)
            except Exception as e:
                st.error(f"The agent failed: {e}. Is Ollama running at {OLLAMA_URL}?")
                st.stop()
        st.markdown(out["messages"][-1].content)
        st.caption(f"Tool: `{out['tool']}` · input: `{out['tool_input']}` · router attempts: {out['attempts']}")
        if out["tool"] == "search_docs":
            with st.expander("Retrieved chunks"):
                st.text(out["tool_output"])
