
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import cast, Any
from sklearn.metrics import confusion_matrix, classification_report

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
CLASSES = ["cloudy", "foggy", "rainy", "shine", "sunrise"]


def collect_predictions(model, dataset):

    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix(model, dataset, output_path: Path, title: str = "Confusion Matrix"):

    y_true, y_pred = collect_predictions(model, dataset)

    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.4,
        linecolor="#cccccc",
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved -> {output_path}")
    return y_true, y_pred




def save_class_metrics(y_true, y_pred, output_path: Path, title: str = "Per-class Metrics"):
 
    report = cast(dict[str, Any], classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    ))

    metric_names = ["precision", "recall", "f1-score"]
    x      = np.arange(len(CLASSES))
    width  = 0.25
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        values = [report[cls][metric] for cls in CLASSES]
        bars   = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                        color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#333333",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Class metrics chart saved -> {output_path}")




def save_class_distribution(train_dir: Path, val_dir: Path, test_dir: Path,
                             output_path: Path):

    splits = {"Train": train_dir, "Validation": val_dir, "Test": test_dir}
    counts = {
        split: [len(list((d / cls).iterdir())) for cls in CLASSES]
        for split, d in splits.items()
    }

    x      = np.arange(len(CLASSES))
    width  = 0.25
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (split, color) in enumerate(zip(splits, colors)):
        ax.bar(x + i * width, counts[split], width, label=split,
               color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylabel("Number of images", fontsize=11)
    ax.set_title("Dataset class distribution across splits",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Class distribution chart saved -> {output_path}")
