"""Evaluate every model x retrieval mode on eval/questions.jsonl in two passes.

Pass 1 (agent): all models x modes answer all questions -> tool accuracy, valid-JSON rate, P50/P95 latency.
Pass 2 (ragas): the JUDGE model's runs only, on a subset of doc questions -> context precision, recall, faithfulness.
Grading reuses the chunks saved in pass 1, so only the judge is loaded on the GPU while grading.

Both passes resume from their CSVs; delete eval/results.csv, details.csv and ragas_scores.csv for a fresh run.

Usage: python eval/run_eval.py          (both passes)
       python eval/run_eval.py agent    (pass 1 only)
       python eval/run_eval.py ragas    (pass 2 only; resumes from eval/ragas_scores.csv)
       python eval/run_eval.py ragas phi4-mini llama3.2:3b   (grade other models' answers, same judge)
"""
import csv
import json
import math
import statistics
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from agent import ask, build_graph
from embeddings import EMBEDDING_MODEL_NAME
from plot_results import plot

import langchain_community.llms
_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
_vertex.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules[_vertex.__name__] = _vertex
if not hasattr(langchain_community.llms, "VertexAI"):
    langchain_community.llms.VertexAI = type("VertexAI", (), {})
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

MODELS = ["qwen3:4b-instruct", "gemma4:e4b", "phi4-mini", "llama3.2:3b"]
MODES = ["dense", "hybrid", "hybrid_rerank"]
JUDGE = "qwen3:4b-instruct"  # one fixed judge so ragas scores are comparable
RAGAS_QUESTIONS = 36         # doc questions graded per mode, spread evenly over the doc set (36 = all)
EVAL = ROOT / "eval"
RESULTS, DETAILS, SCORES = EVAL / "results.csv", EVAL / "details.csv", EVAL / "ragas_scores.csv"
METRIC_COLUMNS = {"llm_context_precision_with_reference": "context_precision",
                  "context_recall": "context_recall", "faithfulness": "faithfulness"}

def percentile(values, p):
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1] if len(values) > 1 else values[0]

def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []

def write_csv(path, data):
    keys = list(dict.fromkeys(k for d in data for k in d))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(data)

def ragas_subset(questions):
    docs = [q["question"] for q in questions if q["tool"] == "search_docs"]
    n = min(RAGAS_QUESTIONS, len(docs))
    return {docs[round(i * (len(docs) - 1) / max(n - 1, 1))] for i in range(n)}

def agent_pass(questions):
    graph = build_graph(MemorySaver())  # in-memory: eval questions must not share chat history
    # Resume: keep finished combos from a previous (interrupted) run
    results, details = read_csv(RESULTS), read_csv(DETAILS)
    finished = {(r["model"], r["mode"]) for r in results}
    details = [d for d in details if (d["model"], d["mode"]) in finished]
    for model in MODELS:
        for mode in MODES:
            if (model, mode) in finished:
                print(f"=== {model} / {mode}: already done, skipping ===", flush=True)
                continue
            print(f"\n=== {model} / {mode} ({len(questions)} questions) ===", flush=True)
            runs = []
            for q in questions:
                start = time.perf_counter()
                try:
                    out = ask(graph, q["question"], str(uuid.uuid4()), model, mode)
                    tool, answer, json_valid, attempts = out["tool"], out["messages"][-1].content, out["json_valid"], out["attempts"]
                    tool_input, contexts = out["tool_input"], out["tool_output"] if out["tool"] == "search_docs" else ""
                except Exception as e:  # one broken run must not kill a multi-hour eval
                    tool, answer, json_valid, attempts, tool_input, contexts = "error", f"ERROR: {e}", False, 0, "", ""
                latency = time.perf_counter() - start
                runs.append({"model": model, "mode": mode, "question": q["question"], "expected_tool": q["tool"],
                             "tool": tool, "correct_tool": tool == q["tool"], "json_valid": json_valid,
                             "attempts": attempts, "latency_s": round(latency, 2), "tool_input": tool_input,
                             "expected_answer": q["expected_answer"], "answer": answer, "contexts": contexts})
                mark = "OK " if tool == q["tool"] else "BAD"
                print(f"  {mark} {tool:13} {latency:5.1f}s  {q['question'][:60]}", flush=True)
            latencies = [r["latency_s"] for r in runs]
            row = {"model": model, "mode": mode, "n": len(runs),
                   "tool_accuracy": round(sum(r["correct_tool"] for r in runs) / len(runs), 3),
                   "valid_json_rate": round(sum(r["json_valid"] for r in runs) / len(runs), 3),
                   "latency_p50_s": round(statistics.median(latencies), 2),
                   "latency_p95_s": round(percentile(latencies, 95), 2)}
            print(" ", row, flush=True)
            results.append(row)
            details += runs
            # Write after every combo so a crash late in the run keeps what finished
            write_csv(RESULTS, results)
            write_csv(DETAILS, details)

def ragas_pass(questions, models):
    subset = ragas_subset(questions)
    todo = [r for r in read_csv(DETAILS) if r["model"] in models and r["question"] in subset
            and r["expected_tool"] == "search_docs" and r["tool"] == "search_docs"]
    scores = read_csv(SCORES)
    for s in scores:
        s["model"] = s.get("model") or JUDGE  # rows from before the model column were all the judge's own runs
    done = {(s["model"], s["mode"], s["question"]) for s in scores}
    todo = [r for r in todo if (r["model"], r["mode"], r["question"]) not in done]
    print(f"\n=== ragas: {len(todo)} to grade for {models} with judge {JUDGE} ({len(done)} already done) ===", flush=True)

    # Free the GPU for the judge: agent models and the embedder are not needed while grading
    for model in [*MODELS, EMBEDDING_MODEL_NAME]:
        if model != JUDGE:
            subprocess.run(["ollama", "stop", model], capture_output=True)

    judge = LangchainLLMWrapper(ChatOllama(model=JUDGE, temperature=0, reasoning=False))
    metrics = [LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness()]
    for i, r in enumerate(todo, 1):
        start = time.perf_counter()
        sample = {"user_input": r["question"], "retrieved_contexts": r["contexts"].split("\n---\n"),
                  "response": r["answer"], "reference": r["expected_answer"]}
        try:
            # One sample at a time so every graded question is saved; one worker because the GPU runs requests serially anyway
            df = evaluate(EvaluationDataset.from_list([sample]), metrics=metrics, llm=judge,
                          run_config=RunConfig(timeout=600, max_workers=1), show_progress=False).to_pandas()
            row = {name: round(float(df[col].iloc[0]), 3) for col, name in METRIC_COLUMNS.items()}
        except Exception as e:
            print(f"  ragas failed: {e}", flush=True)
            row = {name: float("nan") for name in METRIC_COLUMNS.values()}
        scores.append({"model": r["model"], "mode": r["mode"], "question": r["question"], **row})
        write_csv(SCORES, scores)
        print(f"  [{i}/{len(todo)}] {r['model']:17} {r['mode']:13} {time.perf_counter() - start:5.0f}s  {row}  {r['question'][:50]}", flush=True)

    # Average per model x mode (NaN = judge reply unparseable, left out) and attach to the rows in results.csv
    results = read_csv(RESULTS)
    for res in results:
        mode_scores = [s for s in scores if s["model"] == res["model"] and s["mode"] == res["mode"]]
        if not mode_scores:
            continue
        for name in METRIC_COLUMNS.values():
            vals = [float(s[name]) for s in mode_scores if not math.isnan(float(s[name]))]
            res[name] = round(statistics.mean(vals), 3) if vals else ""
        res["ragas_n"] = len(mode_scores)
    write_csv(RESULTS, results)
    print("\n" + "\n".join(str(r) for r in results if r.get("ragas_n")), flush=True)

def main():
    questions = [json.loads(line) for line in open(EVAL / "questions.jsonl", encoding="utf-8")]
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("all", "agent"):
        agent_pass(questions)
    if step in ("all", "ragas"):
        ragas_pass(questions, sys.argv[2:] or [JUDGE])
    plot()

if __name__ == "__main__":
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5.5
    sample_qs = [{"question": f"q{i}", "tool": "search_docs"} for i in range(40)]
    assert len(ragas_subset(sample_qs)) == 36 and {"q0", "q39"} <= ragas_subset(sample_qs)
    main()
