from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_figures(A, quality, out_dir):
    d = Path(out_dir) / "figures"; d.mkdir(parents=True, exist_ok=True)
    paths = {}
    cc = quality.get("category_counts", {})
    if cc:
        f, ax = plt.subplots(figsize=(7, 4)); ax.barh(list(cc), list(cc.values())); ax.invert_yaxis()
        ax.set_xlabel("Sentences"); ax.set_title("Corpus distribution by bias category (ground truth)"); f.tight_layout()
        paths["Fig1_corpus_distribution.png"] = d / "Fig1_corpus_distribution.png"; f.savefig(paths["Fig1_corpus_distribution.png"], dpi=200); plt.close(f)
    cm = A["confusion"]
    f, ax = plt.subplots(figsize=(4.5, 4)); ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks([0, 1], ["Biased", "Not biased"]); ax.set_yticks([0, 1], ["Biased", "Not biased"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    mx = cm.values.max() if cm.values.max() else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", color="white" if cm.values[i, j] > mx / 2 else "black", fontsize=14)
    f.tight_layout(); p = d / "Fig2_confusion_matrix.png"; f.savefig(p, dpi=200); plt.close(f); paths[p.name] = p
    b = A["binary"]; names = ["Accuracy", "Precision", "Recall", "F1"]
    vals = [b[n] if b[n] is not None else 0 for n in names]
    f, ax = plt.subplots(figsize=(5, 4)); bars = ax.bar(names, vals); ax.set_ylim(0, 1.05); ax.set_title("Detection metrics")
    for bar, n in zip(bars, names):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .01, "undef." if b[n] is None else f"{b[n]:.3f}", ha="center")
    f.tight_layout(); p = d / "Fig3_detection_metrics.png"; f.savefig(p, dpi=200); plt.close(f); paths[p.name] = p
    ca = A["category"]; ca = ca[ca.Category != "(Not biased - no category)"]
    if len(ca):
        f, ax = plt.subplots(figsize=(8, 4.5)); x = range(len(ca)); w = 0.4
        ax.bar([i - w / 2 for i in x], ca.Binary_recall.fillna(0), w, label="Binary recall (detected as biased)")
        ax.bar([i + w / 2 for i in x], ca["Category_accuracy(of_evaluated)"].fillna(0), w, label="Category accuracy (of evaluated)")
        ax.set_xticks(list(x), [c if len(c) < 22 else c[:20] + "…" for c in ca.Category], rotation=30, ha="right"); ax.set_ylim(0, 1.05)
        ax.legend(); ax.set_title("Detection performance by category"); f.tight_layout()
        p = d / "Fig4_category_performance.png"; f.savefig(p, dpi=200); plt.close(f); paths[p.name] = p
    e = A["expert"]
    if e is not None and len(e["criterion"]):
        c = e["criterion"]; f, ax = plt.subplots(figsize=(7, 4))
        ax.barh(c.Criterion, c.Mean, xerr=c["SD(sample)"].fillna(0)); ax.invert_yaxis(); ax.set_xlabel("Mean rating (±SD)"); ax.set_title("Expert validation by criterion"); f.tight_layout()
        p = d / "Fig5_expert_validation.png"; f.savefig(p, dpi=200); plt.close(f); paths[p.name] = p
    return paths
