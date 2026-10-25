"""Chart eval/results.csv: one small panel per metric, bars per model, colored by retrieval mode."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL = Path(__file__).parent
MODE_COLORS = {"dense": "#2a78d6", "hybrid": "#eb6834", "hybrid_rerank": "#1baf7a"}  # categorical slots 1-3
METRICS = [("tool_accuracy", "Tool-selection accuracy"), ("valid_json_rate", "Valid-JSON rate (1st try)"),
           ("latency_p50_s", "Latency P50 (s)"), ("latency_p95_s", "Latency P95 (s)"),
           ("context_precision", "Context precision"), ("context_recall", "Context recall"),
           ("faithfulness", "Faithfulness")]

def value(rows, model, mode, key):
    row = next((r for r in rows if r["model"] == model and r["mode"] == mode), None)
    return float(row[key]) if row and row.get(key) else float("nan")

def plot(csv_path=EVAL / "results.csv", out=EVAL / "results.png"):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    models = list(dict.fromkeys(r["model"] for r in rows))
    modes = [m for m in MODE_COLORS if any(r["mode"] == m for r in rows)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    width = 0.8 / len(modes)
    for ax, (key, title) in zip(axes.flat, METRICS):
        for j, mode in enumerate(modes):
            xs = [i + j * width for i in range(len(models))]
            ax.bar(xs, [value(rows, m, mode, key) for m in models], width * 0.92, color=MODE_COLORS[mode], label=mode)
        ax.set_title(title, fontsize=11, loc="left", color="#222")
        ax.set_xticks([i + width * (len(modes) - 1) / 2 for i in range(len(models))], models, fontsize=8, rotation=15)
        if "latency" not in key:
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#555")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(*axes.flat[0].get_legend_handles_labels(), title="Retrieval mode", loc="center", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")

if __name__ == "__main__":
    plot()
